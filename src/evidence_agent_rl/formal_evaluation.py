"""Gated materialization of formal SFT and RL checkpoint evaluations."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .config_materialization import find_placeholders, load_ready_environment_versions
from .experiment import experiment_config_from_dict, sha256_file
from .review_readiness import validate_double_review_readiness
from .rl_training import validate_adapter_manifest
from .serving import model_server_config_from_dict


FORMAL_EVALUATION_ARM_CONTRACTS = {
    "react_no_rl_no_memory": ("base", "none"),
    "sft_only": ("sft", "evidence_bound_reconstruction"),
    "rl_search_no_memory": ("rl", "none"),
    "rl_raw_replay": ("rl", "raw_replay"),
    "rl_query_guidance": ("rl", "query_guidance"),
    "rl_evidence_bound_reconstruction": ("rl", "evidence_bound_reconstruction"),
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


def load_formal_evaluation_spec(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {
        "version", "epistemic_status", "model", "data", "evaluation", "serving", "arms"
    }
    if not isinstance(raw, dict) or set(raw) != fields or raw.get("version") != 1:
        raise ValueError("formal evaluation spec fields differ from version 1")
    if raw.get("epistemic_status") != "frozen_formal_evaluation_design_before_gpu_runs":
        raise ValueError("formal evaluation spec epistemic status is invalid")
    model = raw.get("model")
    if not isinstance(model, dict) or set(model) != {"id", "revision", "dtype"}:
        raise ValueError("formal evaluation model identity is invalid")
    arms = raw.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(FORMAL_EVALUATION_ARM_CONTRACTS):
        raise ValueError("formal evaluation spec must contain six model arms exactly once")
    expected_fields = {
        "checkpoint_family", "training_memory_mode", "experiment_id", "service_id",
        "served_model_name", "result_path",
    }
    identities: dict[str, set[str]] = {
        "experiment_id": set(), "service_id": set(), "served_model_name": set(),
        "result_path": set(),
    }
    for arm_id, (family, memory_mode) in FORMAL_EVALUATION_ARM_CONTRACTS.items():
        arm = arms[arm_id]
        if not isinstance(arm, dict) or set(arm) != expected_fields:
            raise ValueError(f"formal evaluation arm fields drifted: {arm_id}")
        if arm["checkpoint_family"] != family or arm["training_memory_mode"] != memory_mode:
            raise ValueError(f"formal evaluation arm semantics drifted: {arm_id}")
        for field, seen in identities.items():
            value = arm[field]
            if not isinstance(value, str) or not value.strip() or value in seen:
                raise ValueError(f"formal evaluation {field} must be unique: {arm_id}")
            seen.add(value)
    evaluation = raw.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("formal evaluation settings are missing")
    if evaluation.get("memory_modes") != [
        "none", "raw_replay", "query_guidance", "evidence_bound_reconstruction"
    ] or evaluation.get("budgets") != [4, 8, 12]:
        raise ValueError("formal evaluation must preserve the frozen 4x3 Harness matrix")
    if evaluation.get("max_gpu_hours") != 2:
        raise ValueError("each formal adapter evaluation must remain capped at two GPU hours")
    return raw


def validate_formal_evaluation_adapter(
    manifest_path: Path,
    *,
    root: Path,
    model_id: str,
    model_revision: str,
    checkpoint_family: str,
    training_memory_mode: str,
) -> int:
    adapter_dir = validate_adapter_manifest(
        manifest_path,
        repo_root=root,
        model_id=model_id,
        model_revision=model_revision,
    )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = raw.get("plan") if isinstance(raw, dict) else None
    if not isinstance(plan, dict):
        raise ValueError("formal evaluation adapter lacks a training plan")
    if checkpoint_family == "sft":
        training = plan.get("training")
        if (
            raw.get("epistemic_status") != "measured_training_run_not_evaluation_result"
            or not isinstance(training, dict)
            or training.get("method") != "lora_sft"
        ):
            raise ValueError("SFT-only evaluation requires a formal SFT adapter")
    else:
        rollout = plan.get("rollout_manifest")
        if (
            raw.get("epistemic_status") != "measured_rl_update_not_evaluation_result"
            or not isinstance(rollout, dict)
            or rollout.get("memory_mode") != training_memory_mode
        ):
            raise ValueError("RL evaluation adapter training memory mode differs from frozen arm")
    try:
        adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        rank = int(adapter_config["r"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("formal evaluation adapter_config lacks LoRA rank") from exc
    if rank not in {1, 8, 16, 32, 64, 128, 256, 320, 512}:
        raise ValueError("formal evaluation LoRA rank is unsupported by vLLM")
    return rank


def materialize_formal_evaluation(
    spec_path: str | Path,
    environment_report_path: str | Path,
    review_readiness_path: str | Path,
    adapter_manifest_path: str | Path | None,
    *,
    arm_id: str,
    repo_root: str | Path,
) -> dict[str, dict[str, Any]]:
    """Return matched evaluation and inert model-service configs for one arm."""

    root = Path(repo_root).resolve()
    spec_file = _inside_repo(spec_path, root, label="formal evaluation spec")
    environment_file = _inside_repo(
        environment_report_path, root, label="environment report"
    )
    review_file = _inside_repo(
        review_readiness_path, root, label="double-review readiness report"
    )
    spec = load_formal_evaluation_spec(spec_file)
    if arm_id not in FORMAL_EVALUATION_ARM_CONTRACTS:
        raise ValueError(f"unknown formal evaluation arm: {arm_id}")
    review_raw = json.loads(review_file.read_text(encoding="utf-8"))
    readiness = validate_double_review_readiness(review_raw, repo_root=root)
    arm = spec["arms"][arm_id]
    model = spec["model"]
    adapter_file: Path | None = None
    rank: int | None = None
    if arm["checkpoint_family"] == "base":
        if adapter_manifest_path is not None:
            raise ValueError("base ReAct evaluation forbids an adapter manifest")
    else:
        if adapter_manifest_path is None:
            raise ValueError("adapter-backed formal evaluation requires an adapter manifest")
        adapter_file = _inside_repo(adapter_manifest_path, root, label="adapter manifest")
        rank = validate_formal_evaluation_adapter(
            adapter_file,
            root=root,
            model_id=str(model["id"]),
            model_revision=str(model["revision"]),
            checkpoint_family=arm["checkpoint_family"],
            training_memory_mode=arm["training_memory_mode"],
        )
    serving_packages = load_ready_environment_versions(
        environment_file,
        required_packages={"torch", "vllm", "transformers", "peft"},
        purpose="serving",
        minimum_versions={"transformers": "4.51.0", "vllm": "0.8.5"},
    )
    artifacts = readiness.artifacts
    reviewed_tasks = Path(artifacts["reviewed_tasks"]["path"]).resolve()
    reviewed_documents = Path(artifacts["reviewed_documents"]["path"]).resolve()
    pages = Path(artifacts["pages"]["path"]).resolve()
    data = deepcopy(spec["data"])
    data.update({
        "tasks": _display(reviewed_tasks, root),
        "documents": _display(reviewed_documents, root),
        "pages": _display(pages, root),
    })
    settings = deepcopy(spec["evaluation"])
    max_gpu_hours = settings.pop("max_gpu_hours")
    runtime_memory_mode = arm["training_memory_mode"]
    checkpoint_kind = "base" if arm["checkpoint_family"] == "base" else "adapter"
    evaluation_config = {
        "version": 1,
        "experiment_id": arm["experiment_id"],
        "stage": "evaluation",
        "result_path": arm["result_path"],
        "model": {
            **deepcopy(model),
            "checkpoint_kind": checkpoint_kind,
            "served_model_name": arm["served_model_name"],
            "adapter_manifest": _display(adapter_file, root) if adapter_file else None,
        },
        "data": data,
        "runtime": {
            "backend": "rocm",
            "max_gpu_hours": max_gpu_hours,
            "max_tool_steps": 12,
            "max_sequence_length": 4096,
            "memory_mode": runtime_memory_mode,
        },
        "training": {"method": "react_evaluation", **settings},
        "provenance": {
            "formal_evaluation_spec": _display(spec_file, root),
            "formal_evaluation_spec_sha256": sha256_file(spec_file),
            "environment_report": _display(environment_file, root),
            "environment_report_sha256": sha256_file(environment_file),
            "review_readiness_report": _display(review_file, root),
            "review_readiness_report_sha256": sha256_file(review_file),
            "checkpoint_family": arm["checkpoint_family"],
            "training_memory_mode": arm["training_memory_mode"],
        },
    }
    if adapter_file is not None:
        evaluation_config["provenance"]["adapter_manifest_sha256"] = sha256_file(adapter_file)
    serving = deepcopy(spec["serving"])
    service_config = {
        "version": 1,
        "service_id": arm["service_id"],
        "model_id": model["id"],
        "model_revision": model["revision"],
        "tokenizer_revision": model["revision"],
        "checkpoint_kind": checkpoint_kind,
        "adapter_manifest": _display(adapter_file, root) if adapter_file else None,
        "served_model_name": arm["served_model_name"],
        **serving,
        "max_lora_rank": rank,
        "package_versions": serving_packages,
    }
    if unresolved := find_placeholders({
        "evaluation_config": evaluation_config, "service_config": service_config
    }):
        raise ValueError(f"formal evaluation materialization contains placeholders: {unresolved}")
    experiment_config_from_dict(evaluation_config)
    # The server loader is authoritative for launch-plan schema and LoRA constraints.
    # The CLI validates the evaluation config after serialization through its loader.
    model_server_config_from_dict(service_config)
    return {"evaluation_config": evaluation_config, "service_config": service_config}
