"""Materialize the four formal RL arms only after measured method selection."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .config_materialization import find_placeholders, load_ready_environment_versions
from .experiment import sha256_file
from .rl_training import validate_adapter_manifest
from .stability import validate_rl_stability_selection


FORMAL_RL_ARM_MEMORY_MODES = {
    "rl_no_memory_train": "none",
    "rl_raw_replay_train": "raw_replay",
    "rl_query_guidance_train": "query_guidance",
    "rl_evidence_bound_train": "evidence_bound_reconstruction",
}


def _inside_repo(path: str | Path, root: Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"{label} must live inside the repository")
    if not resolved.is_file():
        raise ValueError(f"{label} is missing: {resolved}")
    return resolved


def _display(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _ready_environment_versions(path: Path, required: set[str]) -> dict[str, str]:
    return load_ready_environment_versions(
        path,
        required_packages=required,
        purpose="RL",
        minimum_versions={"transformers": "4.51.0"},
    )


def load_formal_rl_arm_spec(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "version", "epistemic_status", "model", "data", "runtime", "training", "arms"
    }
    if not isinstance(raw, dict) or set(raw) != required or raw.get("version") != 1:
        raise ValueError("formal RL arm spec fields differ from version 1")
    if raw.get("epistemic_status") != "frozen_formal_rl_arm_design_before_gpu_runs":
        raise ValueError("formal RL arm spec epistemic status is invalid")
    arms = raw.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(FORMAL_RL_ARM_MEMORY_MODES):
        raise ValueError("formal RL arm spec must contain the four frozen arms exactly once")
    experiment_ids: set[str] = set()
    output_dirs: set[str] = set()
    rollout_manifests: set[str] = set()
    for arm_id, expected_mode in FORMAL_RL_ARM_MEMORY_MODES.items():
        arm = arms[arm_id]
        required_arm = {
            "experiment_id", "memory_mode", "max_gpu_hours", "output_dir", "rollout_manifest"
        }
        if not isinstance(arm, dict) or set(arm) != required_arm:
            raise ValueError(f"formal RL arm fields drifted: {arm_id}")
        if arm.get("memory_mode") != expected_mode:
            raise ValueError(f"formal RL memory mode drifted: {arm_id}")
        if arm.get("max_gpu_hours") != 8:
            raise ValueError(f"formal RL GPU allocation drifted: {arm_id}")
        for field, seen in (
            ("experiment_id", experiment_ids),
            ("output_dir", output_dirs),
            ("rollout_manifest", rollout_manifests),
        ):
            value = arm.get(field)
            if not isinstance(value, str) or not value.strip() or value in seen:
                raise ValueError(f"formal RL arm {field} must be unique: {arm_id}")
            seen.add(value)
    return raw


def materialize_formal_rl_arm(
    spec_path: str | Path,
    stability_selection_path: str | Path,
    environment_report_path: str | Path,
    sft_adapter_manifest_path: str | Path,
    *,
    arm_id: str,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build one immutable ``rl_train`` config from replay-valid measured inputs."""

    root = Path(repo_root).resolve()
    spec_file = _inside_repo(spec_path, root, label="formal RL arm spec")
    selection_file = _inside_repo(
        stability_selection_path, root, label="RL stability selection"
    )
    report_file = _inside_repo(environment_report_path, root, label="environment report")
    adapter_file = _inside_repo(
        sft_adapter_manifest_path, root, label="SFT adapter manifest"
    )
    spec = load_formal_rl_arm_spec(spec_file)
    if arm_id not in FORMAL_RL_ARM_MEMORY_MODES:
        raise ValueError(f"unknown formal RL arm: {arm_id}")
    selection_raw = json.loads(selection_file.read_text(encoding="utf-8"))
    selection = validate_rl_stability_selection(selection_raw, repo_root=root)
    if not selection.ready or selection.selected_method not in {"grpo", "rloo"}:
        raise ValueError("RL stability selection is not ready")

    model = deepcopy(spec["model"])
    validate_adapter_manifest(
        adapter_file,
        repo_root=root,
        model_id=str(model["id"]),
        model_revision=str(model["revision"]),
    )
    adapter_raw = json.loads(adapter_file.read_text(encoding="utf-8"))
    adapter_plan = adapter_raw.get("plan") if isinstance(adapter_raw, dict) else None
    adapter_training = adapter_plan.get("training") if isinstance(adapter_plan, dict) else None
    if (
        adapter_raw.get("epistemic_status") != "measured_training_run_not_evaluation_result"
        or not isinstance(adapter_training, dict)
        or adapter_training.get("method") != "lora_sft"
    ):
        raise ValueError("formal RL arms require an SFT adapter, not an RL adapter")

    required_packages = {"torch", "transformers", "peft", "accelerate", "trackio"}
    package_versions = _ready_environment_versions(report_file, required_packages)
    arm = spec["arms"][arm_id]
    model["adapter_manifest"] = _display(adapter_file, root)
    data = deepcopy(spec["data"])
    data["rollout_manifest"] = arm["rollout_manifest"]
    runtime = deepcopy(spec["runtime"])
    runtime.update({
        "max_gpu_hours": arm["max_gpu_hours"],
        "memory_mode": arm["memory_mode"],
        "training_package_versions": package_versions,
    })
    training = deepcopy(spec["training"])
    training.update({
        "method": selection.selected_method,
        "output_dir": arm["output_dir"],
    })
    result = {
        "version": 1,
        "experiment_id": arm["experiment_id"],
        "stage": "rl_train",
        "model": model,
        "data": data,
        "runtime": runtime,
        "training": training,
        "provenance": {
            "formal_rl_arm_spec": _display(spec_file, root),
            "formal_rl_arm_spec_sha256": sha256_file(spec_file),
            "stability_selection_report": _display(selection_file, root),
            "stability_selection_report_sha256": sha256_file(selection_file),
            "environment_report": _display(report_file, root),
            "environment_report_sha256": sha256_file(report_file),
            "sft_adapter_manifest_sha256": sha256_file(adapter_file),
        },
    }
    if unresolved := find_placeholders(result):
        raise ValueError(f"materialized config still contains placeholders: {unresolved}")
    return result
