"""Auditable multi-round online RL handoff over Harness rollout manifests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .experiment import load_experiment_config, sha256_file
from .models import MemoryMode
from .rl_training import load_rollout_batch_manifest, validate_adapter_manifest


def _fingerprint(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("online RL artifact escapes the repository")
    if not resolved.is_file():
        raise ValueError(f"online RL artifact is missing: {resolved}")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _resolve_fingerprint(raw: object, root: Path, label: str) -> Path:
    if not isinstance(raw, dict) or set(raw) != {"path", "sha256", "bytes"}:
        raise ValueError(f"online RL {label} fingerprint is invalid")
    path = (root / str(raw["path"])).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"online RL {label} escapes the repository")
    if not path.is_file() or sha256_file(path) != raw["sha256"]:
        raise ValueError(f"online RL {label} hash mismatch")
    if path.stat().st_size != raw["bytes"]:
        raise ValueError(f"online RL {label} size mismatch")
    return path


def initialize_online_rl_state(
    *,
    arm_id: str,
    model_id: str,
    model_revision: str,
    memory_mode: str,
    advantage_method: str,
    reward_mode: str,
    initial_adapter_manifest: str | Path,
    max_rounds: int,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    MemoryMode(memory_mode)
    if advantage_method not in {"grpo", "rloo"}:
        raise ValueError("online RL method must be grpo or rloo")
    if reward_mode not in {
        "composite",
        "outcome_only",
        "memory_balanced_composite",
        "success_dominant_memory_balanced_composite",
    }:
        raise ValueError("online RL reward mode is invalid")
    if not isinstance(max_rounds, int) or isinstance(max_rounds, bool) or not 1 <= max_rounds <= 20:
        raise ValueError("online RL max_rounds must be in [1, 20]")
    if not arm_id.strip() or not model_id.strip() or not model_revision.strip():
        raise ValueError("online RL identities must be non-empty")
    initial = Path(initial_adapter_manifest).resolve()
    validate_adapter_manifest(
        initial, repo_root=root, model_id=model_id, model_revision=model_revision
    )
    return {
        "version": 1,
        "epistemic_status": "online_rl_state_no_evaluation_claim",
        "arm_id": arm_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "memory_mode": memory_mode,
        "advantage_method": advantage_method,
        "reward_mode": reward_mode,
        "max_rounds": max_rounds,
        "initial_checkpoint": _fingerprint(initial, root),
        "rounds": [],
    }


def _validate_header(raw: object) -> dict[str, Any]:
    fields = {
        "version", "epistemic_status", "arm_id", "model_id", "model_revision",
        "memory_mode", "advantage_method", "reward_mode", "max_rounds",
        "initial_checkpoint", "rounds",
    }
    if not isinstance(raw, dict) or set(raw) != fields or raw.get("version") != 1:
        raise ValueError("online RL state fields differ from version 1")
    if raw["epistemic_status"] != "online_rl_state_no_evaluation_claim":
        raise ValueError("online RL state epistemic status is invalid")
    MemoryMode(str(raw["memory_mode"]))
    if raw["advantage_method"] not in {"grpo", "rloo"}:
        raise ValueError("online RL state method is invalid")
    if raw["reward_mode"] not in {
        "composite",
        "outcome_only",
        "memory_balanced_composite",
        "success_dominant_memory_balanced_composite",
    }:
        raise ValueError("online RL state reward mode is invalid")
    if not isinstance(raw["max_rounds"], int) or not 1 <= raw["max_rounds"] <= 20:
        raise ValueError("online RL state max_rounds is invalid")
    if not isinstance(raw["rounds"], list) or len(raw["rounds"]) > raw["max_rounds"]:
        raise ValueError("online RL state round count is invalid")
    return raw


def _build_round_record(
    state: dict[str, Any],
    *,
    index: int,
    current_checkpoint: dict[str, Any],
    rollout_manifest_path: Path,
    update_config_path: Path,
    output_adapter_manifest_path: Path,
    root: Path,
) -> dict[str, Any]:
    current_path = _resolve_fingerprint(current_checkpoint, root, "input checkpoint")
    rollout_fp = _fingerprint(rollout_manifest_path, root)
    config_fp = _fingerprint(update_config_path, root)
    output_fp = _fingerprint(output_adapter_manifest_path, root)
    config = load_experiment_config(update_config_path)
    if config["stage"] not in {"rl_smoke", "rl_train"}:
        raise ValueError("online RL round requires an RL update config")
    expected = {
        "id": state["model_id"], "revision": state["model_revision"],
    }
    if any(config["model"].get(key) != value for key, value in expected.items()):
        raise ValueError("online RL update model identity drift")
    if config["runtime"]["memory_mode"] != state["memory_mode"]:
        raise ValueError("online RL update memory mode drift")
    if config["training"]["method"] != state["advantage_method"]:
        raise ValueError("online RL update method drift")
    if config["training"]["reward_mode"] != state["reward_mode"]:
        raise ValueError("online RL update reward mode drift")
    configured_input = (root / config["model"]["adapter_manifest"]).resolve()
    configured_rollouts = (root / config["data"]["rollout_manifest"]).resolve()
    if configured_input != current_path:
        raise ValueError("online RL update does not consume the current checkpoint")
    if configured_rollouts != rollout_manifest_path.resolve():
        raise ValueError("online RL update config points to another rollout manifest")
    rollout, _ = load_rollout_batch_manifest(rollout_manifest_path, repo_root=root)
    rollout_expected = {
        "experiment_id": config["experiment_id"],
        "model_id": state["model_id"],
        "model_revision": state["model_revision"],
        "memory_mode": state["memory_mode"],
        "reward_mode": state["reward_mode"],
        "advantage_method": state["advantage_method"],
    }
    for field, value in rollout_expected.items():
        if getattr(rollout, field) != value:
            raise ValueError(f"online RL rollout {field} drift")
    if rollout.policy_identity.adapter_manifest_sha256 != current_checkpoint["sha256"]:
        raise ValueError("online RL rollout was sampled from a stale checkpoint")
    validate_adapter_manifest(
        output_adapter_manifest_path, repo_root=root,
        model_id=state["model_id"], model_revision=state["model_revision"],
    )
    output_raw = json.loads(output_adapter_manifest_path.read_text(encoding="utf-8"))
    plan = output_raw.get("plan") if isinstance(output_raw, dict) else None
    if not isinstance(plan, dict):
        raise ValueError("online RL output manifest lacks launch plan")
    if plan.get("config_sha256") != config_fp["sha256"]:
        raise ValueError("online RL output was not produced by the update config")
    planned_rollout = plan.get("rollout_manifest")
    if planned_rollout != rollout.to_json():
        raise ValueError("online RL output plan does not bind the consumed rollouts")
    if output_fp["sha256"] == current_checkpoint["sha256"]:
        raise ValueError("online RL update did not produce a new checkpoint manifest")
    return {
        "index": index,
        "experiment_id": config["experiment_id"],
        "input_checkpoint": deepcopy(current_checkpoint),
        "rollout_manifest": rollout_fp,
        "update_config": config_fp,
        "output_checkpoint": output_fp,
        "rollout_entry_sha256s": [entry.sha256 for entry in rollout.entries],
    }


def validate_online_rl_state(
    raw: object, *, repo_root: str | Path
) -> dict[str, Any]:
    state = _validate_header(raw)
    root = Path(repo_root).resolve()
    initial_path = _resolve_fingerprint(state["initial_checkpoint"], root, "initial checkpoint")
    validate_adapter_manifest(
        initial_path, repo_root=root, model_id=state["model_id"],
        model_revision=state["model_revision"],
    )
    current = state["initial_checkpoint"]
    used_rollouts: set[str] = set()
    rebuilt_rounds = []
    for index, record in enumerate(state["rounds"], 1):
        if not isinstance(record, dict):
            raise ValueError("online RL round must be an object")
        rollout_path = _resolve_fingerprint(record.get("rollout_manifest"), root, "rollout manifest")
        config_path = _resolve_fingerprint(record.get("update_config"), root, "update config")
        output_path = _resolve_fingerprint(record.get("output_checkpoint"), root, "output checkpoint")
        rebuilt = _build_round_record(
            state, index=index, current_checkpoint=current,
            rollout_manifest_path=rollout_path, update_config_path=config_path,
            output_adapter_manifest_path=output_path, root=root,
        )
        if rebuilt != record:
            raise ValueError(f"online RL round {index} metadata drift")
        overlap = used_rollouts & set(rebuilt["rollout_entry_sha256s"])
        if overlap:
            raise ValueError("online RL reuses rollout payloads across rounds")
        used_rollouts.update(rebuilt["rollout_entry_sha256s"])
        rebuilt_rounds.append(rebuilt)
        current = rebuilt["output_checkpoint"]
    return state


def append_online_rl_round(
    raw: object,
    *,
    rollout_manifest_path: str | Path,
    update_config_path: str | Path,
    output_adapter_manifest_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    state = deepcopy(validate_online_rl_state(raw, repo_root=repo_root))
    if len(state["rounds"]) >= state["max_rounds"]:
        raise ValueError("online RL state already reached max_rounds")
    current = (
        state["initial_checkpoint"]
        if not state["rounds"] else state["rounds"][-1]["output_checkpoint"]
    )
    record = _build_round_record(
        state, index=len(state["rounds"]) + 1, current_checkpoint=current,
        rollout_manifest_path=Path(rollout_manifest_path).resolve(),
        update_config_path=Path(update_config_path).resolve(),
        output_adapter_manifest_path=Path(output_adapter_manifest_path).resolve(),
        root=Path(repo_root).resolve(),
    )
    used = {
        digest for round_record in state["rounds"]
        for digest in round_record["rollout_entry_sha256s"]
    }
    if used & set(record["rollout_entry_sha256s"]):
        raise ValueError("online RL reuses rollout payloads across rounds")
    state["rounds"].append(record)
    validate_online_rl_state(state, repo_root=repo_root)
    return state


def online_rl_next_round_plan(raw: object, *, repo_root: str | Path) -> dict[str, Any]:
    state = validate_online_rl_state(raw, repo_root=repo_root)
    current = (
        state["initial_checkpoint"]
        if not state["rounds"] else state["rounds"][-1]["output_checkpoint"]
    )
    return {
        "arm_id": state["arm_id"],
        "next_round": len(state["rounds"]) + 1,
        "rounds_remaining": state["max_rounds"] - len(state["rounds"]),
        "current_checkpoint": current,
        "required_memory_mode": state["memory_mode"],
        "required_advantage_method": state["advantage_method"],
        "required_reward_mode": state["reward_mode"],
        "must_collect_new_rollouts": True,
    }
