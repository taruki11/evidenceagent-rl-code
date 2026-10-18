"""Reproducible experiment manifests and non-launching readiness gates."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .curation import load_page_records, validate_dataset_quality_gate
from .environment import probe_rocm_environment
from .gpu_budget import load_gpu_budget_ledger, reservation_blocker
from .io import load_documents, load_tasks


PLACEHOLDER_MARKERS = ("REPLACE_", "PLACEHOLDER", "MODEL_ID_REQUIRED")


@dataclass(frozen=True)
class ArtifactFingerprint:
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class CodeFingerprint:
    git_commit: str | None
    git_dirty: bool
    tree_sha256: str


@dataclass(frozen=True)
class PreflightReport:
    experiment_id: str
    launch_ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    manifest: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "launch_ready": self.launch_ready,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "manifest": self.manifest,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: Path, root: Path) -> ArtifactFingerprint:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"required artifact is missing: {path}")
    try:
        display = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        display = str(resolved)
    return ArtifactFingerprint(display, sha256_file(resolved), resolved.stat().st_size)


def hash_code_tree(root: str | Path, paths: Iterable[str] = ("src", "scripts", "skills", "pyproject.toml")) -> str:
    base = Path(root).resolve()
    files: list[Path] = []
    for relative in paths:
        target = base / relative
        if target.is_file():
            files.append(target)
        elif target.is_dir():
            files.extend(
                path
                for path in target.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
            )
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_state(root: Path) -> tuple[str | None, bool]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=root, text=True, capture_output=True, check=False
        )

    commit_result = run("rev-parse", "HEAD")
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    status_result = run("status", "--porcelain")
    dirty = status_result.returncode != 0 or bool(status_result.stdout.strip())
    return commit, dirty


def experiment_config_from_dict(raw: object) -> dict[str, Any]:
    """Validate an already parsed experiment config."""

    if not isinstance(raw, dict):
        raise ValueError("experiment config must be an object")
    required = {"version", "experiment_id", "stage", "model", "data", "runtime", "training"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"experiment config missing fields: {sorted(missing)}")
    if raw["version"] != 1:
        raise ValueError("unsupported experiment config version")
    if raw["stage"] not in {
        "sft_smoke", "sft_train", "rl_smoke", "rl_train", "evaluation"
    }:
        raise ValueError("invalid experiment stage")
    for section in ("model", "data", "runtime", "training"):
        if not isinstance(raw[section], dict):
            raise ValueError(f"{section} must be an object")
    model_required = {"id", "revision", "dtype"}
    if missing_model := model_required - set(raw["model"]):
        raise ValueError(f"model config missing fields: {sorted(missing_model)}")
    if raw["model"]["dtype"] != "bfloat16":
        raise ValueError("reserved AMD experiments currently require bfloat16")
    data_required = {
        "documents", "tasks", "memories", "corpus_manifest", "pages", "split",
        "minimum_review", "quality_gate"
    }
    if missing_data := data_required - set(raw["data"]):
        raise ValueError(f"data config missing fields: {sorted(missing_data)}")
    if raw["data"]["split"] not in {"train", "dev", "test", "ood"}:
        raise ValueError("invalid data split")
    if raw["data"]["minimum_review"] not in {
        "draft", "single_review", "double_review"
    }:
        raise ValueError("invalid minimum_review")
    quality_gate = raw["data"]["quality_gate"]
    if quality_gate not in {
        "auto_provenance",
        "single_review",
        "double_review",
        "phase3_t1_correction_traces",
        "crosssource_correction_traces",
    }:
        raise ValueError("invalid data.quality_gate")
    if quality_gate == "auto_provenance":
        if raw["data"]["split"] != "train" or raw["data"]["minimum_review"] != "draft":
            raise ValueError("auto_provenance requires split=train and minimum_review=draft")
        if not str(raw["data"].get("generation_report", "")).strip():
            raise ValueError("auto_provenance requires data.generation_report")
    elif quality_gate in {"phase3_t1_correction_traces", "crosssource_correction_traces"}:
        # Machine-generated correction traces replayed from the frozen policy's own recorded states.
        # T1's asset supervises ONE action; the cross-source asset supervises a five-step cross-paper
        # trajectory.  Both are deliberately ``draft`` -- not human gold, never recorded as such --
        # so neither is ``auto_provenance`` (the 300-proposal automatic-task route, which demands a
        # generation report) nor a human review tier.  A separate name rather than T1's, because a
        # registration type must say which experiment it belongs to.
        if raw["data"]["split"] != "train" or raw["data"]["minimum_review"] != "draft":
            raise ValueError(
                f"{quality_gate} requires split=train and minimum_review=draft"
            )
        if not str(raw["data"].get("correction_traces_manifest", "")).strip():
            raise ValueError(f"{quality_gate} requires data.correction_traces_manifest")
    elif raw["data"]["minimum_review"] != quality_gate:
        raise ValueError("human quality_gate must equal minimum_review")
    sft_review_gate = raw["data"].get("sft_review_gate")
    if sft_review_gate is not None:
        if raw["stage"] not in {"sft_smoke", "sft_train"}:
            raise ValueError("data.sft_review_gate is allowed only for SFT")
        if sft_review_gate not in {
            "paper_v3_dual_ai_recovery",
            "paper_v3_sequential_multihop_v2",
        }:
            raise ValueError("invalid data.sft_review_gate")
        if quality_gate != "double_review" or raw["data"]["split"] != "train":
            raise ValueError(
                "Paper-v3 dual-AI recovery requires double-reviewed train tasks"
            )
        if not str(raw["data"].get("reviewed_recovery_manifest", "")).strip():
            raise ValueError(
                "Paper-v3 dual-AI recovery requires data.reviewed_recovery_manifest"
            )
    runtime = raw["runtime"]
    runtime_required = {
        "backend",
        "max_gpu_hours",
        "max_tool_steps",
        "max_sequence_length",
        "memory_mode",
    }
    if missing_runtime := runtime_required - set(runtime):
        raise ValueError(f"runtime config missing fields: {sorted(missing_runtime)}")
    if runtime.get("backend") not in {"rocm", "cuda"}:
        raise ValueError("runtime.backend must be rocm or cuda")
    if runtime["memory_mode"] not in {
        "none", "raw_replay", "query_guidance", "evidence_bound_reconstruction"
    }:
        raise ValueError("invalid runtime.memory_mode")
    gpu_hours = runtime.get("max_gpu_hours")
    if not isinstance(gpu_hours, (int, float)) or isinstance(gpu_hours, bool) or not 0 < gpu_hours <= 100:
        raise ValueError("max_gpu_hours must be in (0, 100]")
    if raw["stage"].endswith("_smoke") and gpu_hours > 5:
        raise ValueError("a smoke experiment cannot reserve more than 5 GPU hours")
    if "policy_context_version" in runtime and (
        not isinstance(runtime["policy_context_version"], str)
        or runtime["policy_context_version"] not in {"v1", "v2", "v3", "v4", "v5", "v6"}
    ):
        raise ValueError("runtime.policy_context_version must be v1, v2, v3, v4, v5, or v6")
    if "policy_skill_path" in runtime and (
        not isinstance(runtime["policy_skill_path"], str)
        or not runtime["policy_skill_path"].strip()
    ):
        raise ValueError("runtime.policy_skill_path must be a non-empty string")
    if "chat_template_kwargs" in runtime:
        template_kwargs = runtime["chat_template_kwargs"]
        if not isinstance(template_kwargs, dict) or set(template_kwargs) != {"enable_thinking"}:
            raise ValueError(
                "runtime.chat_template_kwargs must contain only enable_thinking"
            )
        if not isinstance(template_kwargs["enable_thinking"], bool):
            raise ValueError("runtime.chat_template_kwargs.enable_thinking must be boolean")
    if "require_server_token_ids" in runtime and not isinstance(
        runtime["require_server_token_ids"], bool
    ):
        raise ValueError("runtime.require_server_token_ids must be boolean")
    if not isinstance(runtime["max_tool_steps"], int) or not 1 <= runtime["max_tool_steps"] <= 12:
        raise ValueError("max_tool_steps must be an integer in [1, 12]")
    if not isinstance(runtime["max_sequence_length"], int) or not 1024 <= runtime["max_sequence_length"] <= 8192:
        raise ValueError("max_sequence_length must be an integer in [1024, 8192]")
    training = raw["training"]
    seed = training.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("training.seed must be a non-negative integer")
    if raw["stage"] != "evaluation" and (
        not isinstance(training.get("max_steps"), int) or training["max_steps"] <= 0
    ):
        raise ValueError("training.max_steps must be a positive integer")
    if raw["stage"] in {"sft_smoke", "sft_train"} and training.get("method") != "lora_sft":
        raise ValueError("SFT stages require method=lora_sft")
    if raw["stage"] in {"sft_smoke", "sft_train"}:
        if not str(raw["data"].get("sft_dataset", "")).strip():
            raise ValueError("SFT stages require data.sft_dataset")
        validation_fields = {
            "validation_sft_dataset",
            "validation_split",
            "validation_quality_gate",
        }
        present_validation_fields = validation_fields & set(raw["data"])
        if present_validation_fields and present_validation_fields != validation_fields:
            raise ValueError("SFT validation data fields must be supplied together")
        if present_validation_fields:
            if raw["data"]["validation_split"] != "dev":
                raise ValueError("SFT validation_split must be dev")
            if raw["data"]["validation_quality_gate"] not in {
                "single_review", "double_review"
            }:
                raise ValueError("SFT validation requires a reviewed quality gate")
            if not str(raw["data"]["validation_sft_dataset"]).strip():
                raise ValueError("SFT validation_sft_dataset must be non-empty")
        package_versions = runtime.get("training_package_versions")
        required_packages = {
            "torch", "transformers", "datasets", "peft", "accelerate", "trackio"
        }
        if raw["model"].get("quantization_mode") != "bitsandbytes_4bit":
            required_packages.add("trl")
        if not isinstance(package_versions, dict) or set(package_versions) != required_packages:
            raise ValueError(
                "SFT stages require exact runtime.training_package_versions for "
                + ", ".join(sorted(required_packages))
            )
        if any(not isinstance(value, str) or not value.strip() for value in package_versions.values()):
            raise ValueError("runtime.training_package_versions values must be non-empty strings")
        sft_required = {
            "output_dir",
            "learning_rate",
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "lr_scheduler_type",
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "gradient_checkpointing",
            "logging_steps",
            "save_steps",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
        }
        if missing_sft := sft_required - set(training):
            raise ValueError(f"SFT training config missing fields: {sorted(missing_sft)}")
        if present_validation_fields:
            validation_training_fields = {"eval_steps", "per_device_eval_batch_size"}
            if missing_validation := validation_training_fields - set(training):
                raise ValueError(
                    "SFT validation training config missing fields: "
                    f"{sorted(missing_validation)}"
                )
            for field in validation_training_fields:
                if (
                    not isinstance(training[field], int)
                    or isinstance(training[field], bool)
                    or training[field] <= 0
                ):
                    raise ValueError(f"training.{field} must be a positive integer")
            if training["eval_steps"] > training["max_steps"]:
                raise ValueError("training.eval_steps must not exceed training.max_steps")
        if training["lr_scheduler_type"] not in {"linear", "cosine"}:
            raise ValueError("SFT lr_scheduler_type must be linear or cosine")
        for field in ("warmup_ratio", "lora_dropout"):
            value = training[field]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value < 1:
                raise ValueError(f"training.{field} must be in [0, 1)")
        for field in ("learning_rate", "max_grad_norm"):
            value = training[field]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"training.{field} must be positive")
        loss_normalization = training.get("loss_normalization", "token")
        if loss_normalization not in {"token", "action", "hybrid"}:
            raise ValueError(
                "training.loss_normalization must be token, action, or hybrid"
            )
        action_loss_weight = training.get("action_loss_weight", 0.25)
        if (
            not isinstance(action_loss_weight, (int, float))
            or isinstance(action_loss_weight, bool)
            or not 0 <= action_loss_weight <= 1
        ):
            raise ValueError("training.action_loss_weight must be in [0, 1]")
        if not isinstance(training["weight_decay"], (int, float)) or isinstance(training["weight_decay"], bool) or training["weight_decay"] < 0:
            raise ValueError("training.weight_decay must be non-negative")
        for field in (
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "logging_steps",
            "save_steps",
            "lora_r",
            "lora_alpha",
        ):
            if not isinstance(training[field], int) or isinstance(training[field], bool) or training[field] <= 0:
                raise ValueError(f"training.{field} must be a positive integer")
        if not isinstance(training["gradient_checkpointing"], bool):
            raise ValueError("training.gradient_checkpointing must be boolean")
        targets = training["lora_target_modules"]
        if not isinstance(targets, list) or not targets or any(
            not isinstance(item, str) or not item.strip() for item in targets
        ):
            raise ValueError("training.lora_target_modules must be non-empty strings")
        repeat_factor = training.get("memory_trajectory_repeat_factor", 1)
        if (
            not isinstance(repeat_factor, int)
            or isinstance(repeat_factor, bool)
            or not 1 <= repeat_factor <= 16
        ):
            raise ValueError(
                "training.memory_trajectory_repeat_factor must be an integer in [1, 16]"
            )
        decision_repeat = training.get("memory_decision_repeat_factor", 1)
        if (
            not isinstance(decision_repeat, int)
            or isinstance(decision_repeat, bool)
            or not 1 <= decision_repeat <= 16
        ):
            raise ValueError(
                "training.memory_decision_repeat_factor must be an integer in [1, 16]"
            )
        decision_include_base = training.get("memory_decision_include_base_records", False)
        if not isinstance(decision_include_base, bool):
            raise ValueError(
                "training.memory_decision_include_base_records must be boolean"
            )
        if decision_include_base and decision_repeat == 1:
            raise ValueError(
                "training.memory_decision_include_base_records requires decision oversampling"
            )
        transition_repeat = training.get("public_transition_repeat_factor", 1)
        if (
            not isinstance(transition_repeat, int)
            or isinstance(transition_repeat, bool)
            or not 1 <= transition_repeat <= 16
        ):
            raise ValueError(
                "training.public_transition_repeat_factor must be an integer in [1, 16]"
            )
        if sum(value > 1 for value in (repeat_factor, decision_repeat, transition_repeat)) > 1:
            raise ValueError("SFT curricula cannot be combined")
        adapter_manifest = raw["model"].get("adapter_manifest")
        if adapter_manifest is not None and not str(adapter_manifest).strip():
            raise ValueError("SFT model.adapter_manifest must be non-empty when supplied")
    if raw["stage"] in {"rl_smoke", "rl_train"}:
        if training.get("method") not in {"grpo", "rloo"}:
            raise ValueError("RL stages require method grpo or rloo")
        if training.get("credit_assignment_method", training["method"]) not in {
            "grpo",
            "rloo",
            "state_anchored_grpo_v1",
            "state_anchored_grpo_v2",
            "state_anchored_turn_credit_v3",
            "verifier_progress_turn_credit_v4",
            "hybrid_outcome_verifier_progress_v5",
            "multihop_graph_progress_v6",
        }:
            raise ValueError("RL stages require a supported credit_assignment_method")
        if training.get("reward_mode") not in {
            "composite",
            "outcome_only",
            "memory_balanced_composite",
            "success_dominant_memory_balanced_composite",
        }:
            raise ValueError("RL stages require an explicit reward_mode")
        group_size = training.get("group_size")
        if not isinstance(group_size, int) or not 2 <= group_size <= 8:
            raise ValueError("RL group_size must be an integer in [2, 8]")
        if not str(raw["data"].get("rollout_manifest", "")).strip():
            raise ValueError("RL stages require data.rollout_manifest")
        if not str(raw["model"].get("adapter_manifest", "")).strip():
            raise ValueError("RL stages require model.adapter_manifest from SFT")
        package_versions = runtime.get("training_package_versions")
        required_packages = {"torch", "transformers", "peft", "accelerate", "trackio"}
        if not isinstance(package_versions, dict) or set(package_versions) != required_packages:
            raise ValueError(
                "RL stages require exact runtime.training_package_versions for "
                + ", ".join(sorted(required_packages))
            )
        rl_required = {
            "output_dir", "weight_decay", "max_grad_norm",
            "gradient_accumulation_groups", "logging_steps", "save_steps",
        }
        if missing_rl := rl_required - set(training):
            raise ValueError(f"RL training config missing fields: {sorted(missing_rl)}")
        for field in ("gradient_accumulation_groups", "logging_steps", "save_steps"):
            if not isinstance(training[field], int) or isinstance(training[field], bool) or training[field] <= 0:
                raise ValueError(f"training.{field} must be a positive integer")
        for field in ("learning_rate", "max_grad_norm"):
            value = training[field]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"training.{field} must be positive")
        if not isinstance(training["weight_decay"], (int, float)) or isinstance(training["weight_decay"], bool) or training["weight_decay"] < 0:
            raise ValueError("training.weight_decay must be non-negative")
    if raw["stage"] == "evaluation":
        if quality_gate == "auto_provenance":
            raise ValueError("evaluation stage requires a human review quality gate")
        if training.get("method") != "react_evaluation":
            raise ValueError("evaluation requires method=react_evaluation")
        checkpoint_kind = raw["model"].get("checkpoint_kind")
        if checkpoint_kind not in {"base", "adapter"}:
            raise ValueError("evaluation requires model.checkpoint_kind=base or adapter")
        adapter_manifest = raw["model"].get("adapter_manifest")
        if checkpoint_kind == "adapter" and not str(adapter_manifest or "").strip():
            raise ValueError("adapter evaluation requires model.adapter_manifest")
        if checkpoint_kind == "base" and adapter_manifest is not None:
            raise ValueError("base evaluation forbids model.adapter_manifest")
        evaluation_required = {
            "memory_modes", "budgets", "control_memory_mode", "control_budget",
            "temperature", "top_p", "max_action_tokens", "max_decisions",
        }
        if missing_evaluation := evaluation_required - set(training):
            raise ValueError(
                f"evaluation config missing fields: {sorted(missing_evaluation)}"
            )
        modes = training["memory_modes"]
        allowed_modes = {"none", "raw_replay", "query_guidance", "evidence_bound_reconstruction"}
        if not isinstance(modes, list) or set(modes) != allowed_modes or len(modes) != 4:
            raise ValueError("evaluation memory_modes must contain each registered mode once")
        budgets = training["budgets"]
        if not isinstance(budgets, list) or not budgets or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in budgets
        ) or len(set(budgets)) != len(budgets):
            raise ValueError("evaluation budgets must be unique positive integers")
        if training["control_memory_mode"] not in modes or training["control_budget"] not in budgets:
            raise ValueError("evaluation control must be present in the condition matrix")
        if not isinstance(training["temperature"], (int, float)) or training["temperature"] < 0:
            raise ValueError("evaluation temperature must be non-negative")
        if not isinstance(training["top_p"], (int, float)) or not 0 < training["top_p"] <= 1:
            raise ValueError("evaluation top_p must be in (0, 1]")
        for field in ("max_action_tokens", "max_decisions"):
            if not isinstance(training[field], int) or isinstance(training[field], bool) or training[field] <= 0:
                raise ValueError(f"evaluation {field} must be a positive integer")
        if gpu_hours > 5:
            raise ValueError("an evaluation smoke cannot reserve more than 5 GPU hours")
    return raw


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    return experiment_config_from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def _placeholder_paths(value: object, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, str) and any(marker in value for marker in PLACEHOLDER_MARKERS):
        found.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            found.extend(_placeholder_paths(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_placeholder_paths(child, f"{path}[{index}]"))
    return found


def build_preflight_report(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    probe_torch: bool = False,
    probe_rocm: bool = False,
    budget_ledger_path: str | Path | None = None,
    reservation_config_path: str | Path | None = None,
) -> PreflightReport:
    root = Path(repo_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_experiment_config(config_file)
    config_fingerprint = fingerprint_file(config_file, root)
    blockers: list[str] = []
    warnings: list[str] = []

    placeholders = _placeholder_paths(config)
    if placeholders:
        blockers.append(f"unresolved placeholders: {placeholders}")
    model_revision = str(config["model"].get("revision", "")).strip()
    if model_revision.casefold() in {"", "main", "latest"}:
        blockers.append("model revision must be an immutable revision, not main/latest")

    data = config["data"]
    artifact_keys = ["documents", "tasks", "memories", "corpus_manifest", "pages"]
    if config["stage"] in {"sft_smoke", "sft_train"}:
        artifact_keys.append("sft_dataset")
        if data.get("sft_review_gate") in {
            "paper_v3_dual_ai_recovery",
            "paper_v3_sequential_multihop_v2",
        }:
            artifact_keys.append("reviewed_recovery_manifest")
        if "validation_sft_dataset" in data:
            artifact_keys.append("validation_sft_dataset")
    if data["quality_gate"] == "auto_provenance":
        artifact_keys.append("generation_report")
    if data["quality_gate"] == "phase3_t1_correction_traces":
        artifact_keys.append("correction_traces_manifest")
    if config["stage"] in {"rl_smoke", "rl_train"}:
        artifact_keys.append("rollout_manifest")
    artifact_paths = {key: (root / str(data[key])).resolve() for key in artifact_keys}
    provenance = config.get("provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("environment_report"), str):
        artifact_paths["environment_report"] = (
            root / provenance["environment_report"]
        ).resolve()
    if isinstance(provenance, dict):
        for artifact_key, provenance_key in (
            ("formal_rl_arm_spec", "formal_rl_arm_spec"),
            ("stability_selection_report", "stability_selection_report"),
            ("formal_evaluation_spec", "formal_evaluation_spec"),
            ("review_readiness_report", "review_readiness_report"),
        ):
            value = provenance.get(provenance_key)
            if isinstance(value, str):
                artifact_paths[artifact_key] = (root / value).resolve()
    if (
        config["stage"] in {"sft_smoke", "sft_train"}
        and config["model"].get("adapter_manifest") is not None
    ) or config["stage"] in {"rl_smoke", "rl_train"} or (
        config["stage"] == "evaluation"
        and config["model"].get("checkpoint_kind") == "adapter"
    ):
        artifact_paths["adapter_manifest"] = (
            root / str(config["model"]["adapter_manifest"])
        ).resolve()
    artifacts: dict[str, dict[str, Any]] = {}
    for key, path in artifact_paths.items():
        try:
            artifacts[key] = asdict(fingerprint_file(path, root))
        except ValueError as exc:
            blockers.append(f"data artifact failed: {exc}")
    if isinstance(provenance, dict) and "environment_report" in artifact_paths:
        environment_artifact = artifacts.get("environment_report")
        if environment_artifact and environment_artifact["sha256"] != provenance.get(
            "environment_report_sha256"
        ):
            blockers.append("environment report hash does not match config provenance")
        template_path = provenance.get("materialized_from_template")
        if isinstance(template_path, str):
            template_file = (root / template_path).resolve()
            try:
                template_fingerprint = asdict(fingerprint_file(template_file, root))
                artifacts["materialized_template"] = template_fingerprint
                if template_fingerprint["sha256"] != provenance.get("template_sha256"):
                    blockers.append("materialization template hash does not match config provenance")
            except ValueError as exc:
                blockers.append(f"materialization template failed: {exc}")
    if isinstance(provenance, dict):
        for artifact_key, hash_key in (
            ("formal_rl_arm_spec", "formal_rl_arm_spec_sha256"),
            ("stability_selection_report", "stability_selection_report_sha256"),
            ("adapter_manifest", "sft_adapter_manifest_sha256"),
            ("formal_evaluation_spec", "formal_evaluation_spec_sha256"),
            ("review_readiness_report", "review_readiness_report_sha256"),
            ("adapter_manifest", "adapter_manifest_sha256"),
        ):
            artifact = artifacts.get(artifact_key)
            expected = provenance.get(hash_key)
            if hash_key in provenance and artifact is not None and artifact["sha256"] != expected:
                blockers.append(f"{artifact_key} hash does not match config provenance")
        selection_path = artifact_paths.get("stability_selection_report")
        if selection_path is not None and selection_path.is_file():
            try:
                from .stability import validate_rl_stability_selection

                selection_raw = json.loads(selection_path.read_text(encoding="utf-8"))
                selection = validate_rl_stability_selection(selection_raw, repo_root=root)
                if not selection.ready:
                    blockers.append("RL stability selection is not ready")
                elif config["training"].get("method") != selection.selected_method:
                    blockers.append("RL method differs from measured stability selection")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                blockers.append(f"RL stability selection failed: {exc}")
        spec_path = artifact_paths.get("formal_rl_arm_spec")
        if spec_path is not None and spec_path.is_file():
            try:
                from .formal_rl import load_formal_rl_arm_spec

                arm_spec = load_formal_rl_arm_spec(spec_path)
                matching = [
                    arm for arm in arm_spec["arms"].values()
                    if arm["experiment_id"] == config["experiment_id"]
                ]
                if len(matching) != 1:
                    blockers.append("formal RL config is not a frozen arm identity")
                elif matching[0]["memory_mode"] != config["runtime"]["memory_mode"]:
                    blockers.append("formal RL memory mode differs from frozen arm")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                blockers.append(f"formal RL arm spec failed: {exc}")
        review_path = artifact_paths.get("review_readiness_report")
        if review_path is not None and review_path.is_file():
            try:
                from .review_readiness import validate_double_review_readiness

                review_raw = json.loads(review_path.read_text(encoding="utf-8"))
                validate_double_review_readiness(review_raw, repo_root=root)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                blockers.append(f"double-review readiness failed: {exc}")
        evaluation_spec_path = artifact_paths.get("formal_evaluation_spec")
        if evaluation_spec_path is not None and evaluation_spec_path.is_file():
            try:
                from .formal_evaluation import (
                    load_formal_evaluation_spec,
                    validate_formal_evaluation_adapter,
                )

                evaluation_spec = load_formal_evaluation_spec(evaluation_spec_path)
                matching = [
                    arm for arm in evaluation_spec["arms"].values()
                    if arm["experiment_id"] == config["experiment_id"]
                ]
                if len(matching) != 1:
                    blockers.append("formal evaluation config is not a frozen arm identity")
                else:
                    arm = matching[0]
                    if arm["training_memory_mode"] != config["runtime"]["memory_mode"]:
                        blockers.append("formal evaluation memory mode differs from frozen arm")
                    if arm["result_path"] != config.get("result_path"):
                        blockers.append("formal evaluation result path differs from frozen arm")
                    if provenance.get("checkpoint_family") != arm["checkpoint_family"]:
                        blockers.append("formal evaluation checkpoint family differs from frozen arm")
                    adapter_path = artifact_paths.get("adapter_manifest")
                    if adapter_path is not None and adapter_path.is_file():
                        validate_formal_evaluation_adapter(
                            adapter_path,
                            root=root,
                            model_id=str(config["model"]["id"]),
                            model_revision=str(config["model"]["revision"]),
                            checkpoint_family=str(arm["checkpoint_family"]),
                            training_memory_mode=str(arm["training_memory_mode"]),
                        )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                blockers.append(f"formal evaluation spec failed: {exc}")
    tasks = (
        tuple(
            task
            for task in load_tasks(artifact_paths["tasks"])
            if task.split == data["split"]
        )
        if artifact_paths["tasks"].is_file()
        else ()
    )
    if not tasks:
        blockers.append(f"no tasks exist in requested split {data['split']}")
    elif artifact_paths["documents"].is_file() and artifact_paths["pages"].is_file():
        try:
            quality_tasks = tasks
            if data.get("sft_review_gate") == "paper_v3_sequential_multihop_v2":
                quality_tasks = tuple(
                    task for task in tasks if task.review_status != "draft"
                )
            validate_dataset_quality_gate(
                quality_tasks,
                load_documents(artifact_paths["documents"]),
                load_page_records(artifact_paths["pages"]),
                gate=str(data["quality_gate"]),
                split=str(data["split"]),
            )
        except ValueError as exc:
            blockers.append(f"dataset quality gate failed: {exc}")
    if data["quality_gate"] == "auto_provenance" and artifact_paths[
        "generation_report"
    ].is_file():
        try:
            generation_report = json.loads(
                artifact_paths["generation_report"].read_text(encoding="utf-8")
            )
            if generation_report.get("epistemic_status") != (
                "automatic_train_proposals_not_human_reviewed"
            ):
                blockers.append("auto generation report has invalid epistemic status")
            if int(generation_report.get("generated", 0)) < 300:
                blockers.append("auto generation report contains fewer than 300 proposals")
            if int(generation_report.get("materialized_train_tasks", -1)) != len(tasks):
                blockers.append("auto generation report task count does not match selected train data")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"auto generation report failed: {exc}")
    if config["stage"] in {"sft_smoke", "sft_train"} and artifact_paths["sft_dataset"].is_file():
        try:
            from .sft_training import (
                load_and_audit_sft_dataset,
                validate_sft_paper_disjointness,
            )

            load_and_audit_sft_dataset(
                artifact_paths["sft_dataset"],
                expected_split=str(data["split"]),
                quality_gate=str(data.get("sft_review_gate", data["quality_gate"])),
                expected_tasks=len(tasks),
            )
            validation_path = artifact_paths.get("validation_sft_dataset")
            if validation_path is not None and validation_path.is_file():
                load_and_audit_sft_dataset(
                    validation_path,
                    expected_split=str(data["validation_split"]),
                    quality_gate=str(data["validation_quality_gate"]),
                )
                validate_sft_paper_disjointness(
                    artifact_paths["sft_dataset"], validation_path
                )
        except ValueError as exc:
            blockers.append(f"SFT dataset gate failed: {exc}")
    recovery_manifest_path = artifact_paths.get("reviewed_recovery_manifest")
    if recovery_manifest_path is not None and recovery_manifest_path.is_file():
        try:
            recovery_manifest = json.loads(
                recovery_manifest_path.read_text(encoding="utf-8")
            )
            if data.get("sft_review_gate") == "paper_v3_sequential_multihop_v2":
                draft_task_ids = {
                    task.task_id for task in tasks if task.review_status == "draft"
                }
                if (
                    recovery_manifest.get("epistemic_status")
                    != "dual_ai_reviewed_sequential_multihop_v2_training_data_not_model_result"
                    or recovery_manifest.get("candidate_sha256")
                    != data.get("reviewed_candidate_sha256")
                    or recovery_manifest.get("tasks_sha256")
                    != sha256_file(artifact_paths["tasks"])
                    or recovery_manifest.get("sft_dataset_sha256")
                    != sha256_file(artifact_paths["sft_dataset"])
                    or int(recovery_manifest.get("accepted_candidates", 0)) != 33
                    or int(recovery_manifest.get("accepted_supervision_candidates", 0)) != 25
                    or int(recovery_manifest.get("held_out_canary_candidates", 0)) != 8
                    or set(recovery_manifest.get("supervision_task_ids", []))
                    != draft_task_ids
                    or len(set(recovery_manifest.get("reviewer_ids", []))) != 2
                    or recovery_manifest.get("incremental_training_source_disjoint") is not True
                    or recovery_manifest.get("gpu_launch_allowed") is not True
                ):
                    blockers.append("Paper-v3 sequential multi-hop v2 manifest identity drift")
            elif (
                recovery_manifest.get("epistemic_status")
                != "dual_ai_reviewed_frontier_recovery_training_data_not_model_result"
                or recovery_manifest.get("output") != data.get("sft_dataset")
                or recovery_manifest.get("output_sha256")
                != sha256_file(artifact_paths["sft_dataset"])
                or recovery_manifest.get("output_review_status") != "draft"
                or int(recovery_manifest.get("accepted_candidate_count", 0)) != 32
                or len(set(recovery_manifest.get("reviewer_ids", []))) < 2
            ):
                blockers.append("Paper-v3 reviewed recovery manifest identity drift")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            blockers.append(f"Paper-v3 reviewed recovery manifest failed: {exc}")
    if config["stage"] in {"sft_smoke", "sft_train", "rl_smoke", "rl_train", "evaluation"}:
        if "adapter_manifest" in artifact_paths and artifact_paths["adapter_manifest"].is_file():
            try:
                from .rl_training import validate_adapter_manifest

                validate_adapter_manifest(
                    artifact_paths["adapter_manifest"],
                    repo_root=root,
                    model_id=str(config["model"]["id"]),
                    model_revision=str(config["model"]["revision"]),
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                blockers.append(f"SFT adapter gate failed: {exc}")
        if config["stage"] in {"rl_smoke", "rl_train"} and artifact_paths["rollout_manifest"].is_file():
            try:
                from .rl_training import load_rollout_batch_manifest

                rollout_manifest, batches = load_rollout_batch_manifest(
                    artifact_paths["rollout_manifest"], repo_root=root
                )
                expected_identity = {
                    "experiment_id": str(config["experiment_id"]),
                    "model_id": str(config["model"]["id"]),
                    "model_revision": str(config["model"]["revision"]),
                    "memory_mode": str(config["runtime"]["memory_mode"]),
                    "reward_mode": str(config["training"]["reward_mode"]),
                    "advantage_method": str(
                        config["training"].get(
                            "credit_assignment_method", config["training"]["method"]
                        )
                    ),
                }
                for field, value in expected_identity.items():
                    if getattr(rollout_manifest, field) != value:
                        blockers.append(f"rollout manifest {field} does not match config")
                adapter_sha = sha256_file(artifact_paths["adapter_manifest"])
                if rollout_manifest.policy_identity.adapter_manifest_sha256 != adapter_sha:
                    blockers.append("rollout manifest was not sampled from the configured adapter")
                policy_skill_path = root / "skills" / "evidence-research" / "SKILL.md"
                if (
                    not policy_skill_path.is_file()
                    or rollout_manifest.policy_identity.policy_skill_name != "evidence-research"
                    or rollout_manifest.policy_identity.policy_skill_sha256
                    != sha256_file(policy_skill_path)
                ):
                    blockers.append("rollout manifest policy skill differs from the repository contract")
                task_ids = {task.task_id for task in tasks}
                unknown = sorted({batch.task_id for batch in batches} - task_ids)
                if unknown:
                    blockers.append(f"rollout manifest contains tasks outside selected split: {unknown}")
                expected_status = (
                    "draft_rollout_smoke"
                    if data["quality_gate"] == "auto_provenance"
                    else "review_gated_rollout"
                )
                if rollout_manifest.epistemic_status != expected_status:
                    blockers.append("rollout epistemic status does not match data quality gate")
                needed_groups = int(config["training"]["max_steps"]) * int(
                    config["training"]["gradient_accumulation_groups"]
                )
                if rollout_manifest.audit.files < needed_groups:
                    blockers.append(
                        f"rollout manifest has {rollout_manifest.audit.files} groups; "
                        f"{needed_groups} are required without rollout reuse"
                    )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                blockers.append(f"rollout manifest gate failed: {exc}")

    commit, dirty = _git_state(root)
    if commit is None:
        blockers.append("repository has no committed code revision")
    if dirty:
        blockers.append("repository worktree is dirty")
    code = CodeFingerprint(commit, dirty, hash_code_tree(root))

    ledger_candidate = (
        Path(budget_ledger_path)
        if budget_ledger_path is not None
        else Path("configs/gpu_budget_ledger.json")
    )
    ledger_file = (
        ledger_candidate.resolve()
        if ledger_candidate.is_absolute()
        else (root / ledger_candidate).resolve()
    )
    reservation_config = config
    reservation_fingerprint = config_fingerprint
    if reservation_config_path is not None:
        reservation_candidate = Path(reservation_config_path)
        reservation_file = (
            reservation_candidate.resolve()
            if reservation_candidate.is_absolute()
            else (root / reservation_candidate).resolve()
        )
        reservation_config = load_experiment_config(reservation_file)
        reservation_fingerprint = fingerprint_file(reservation_file, root)
    budget_manifest: dict[str, Any] = {
        "path": str(ledger_file),
        "reservation_experiment_id": str(reservation_config["experiment_id"]),
        "reservation_config_sha256": reservation_fingerprint.sha256,
    }
    try:
        ledger = load_gpu_budget_ledger(ledger_file)
        blocker = reservation_blocker(
            ledger,
            experiment_id=str(reservation_config["experiment_id"]),
            config_sha256=reservation_fingerprint.sha256,
            requested_hours=float(reservation_config["runtime"]["max_gpu_hours"]),
        )
        if blocker:
            blockers.append(blocker)
        budget_manifest.update(
            {
                "total_hours": ledger.total_hours,
                "committed_hours": ledger.committed_hours,
                "remaining_hours": ledger.remaining_hours,
                "overrun_hours": ledger.overrun_hours,
                "reservation_valid": blocker is None,
            }
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        blockers.append(f"GPU budget ledger failed: {exc}")
        budget_manifest["error"] = str(exc)

    environment: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_probe_requested": probe_torch,
        "rocm_probe_requested": probe_rocm,
    }
    backend = str(config["runtime"]["backend"])
    if probe_rocm:
        # Thresholds come from the project's formal-scale defaults unless THIS config lowers them.
        # Such a lowering is a recorded policy decision for one experiment, never a global relaxation:
        # existing configs that say nothing keep the 180 GiB default byte for byte.
        runtime_cfg = config.get("runtime", {}) or {}
        rocm_report = probe_rocm_environment(
            root,
            minimum_gpu_memory_gib=float(runtime_cfg.get("minimum_gpu_memory_gib", 180.0)),
            minimum_system_memory_gib=float(runtime_cfg.get("minimum_system_memory_gib", 180.0)),
            minimum_disk_free_gib=float(runtime_cfg.get("minimum_disk_free_gib", 80.0)),
        )
        environment["rocm"] = rocm_report.to_json()
        blockers.extend(f"ROCm environment: {item}" for item in rocm_report.blockers)
        warnings.extend(f"ROCm environment: {item}" for item in rocm_report.warnings)
    elif probe_torch:
        try:
            import torch  # type: ignore

            environment["torch_version"] = torch.__version__
            environment["torch_hip"] = torch.version.hip
            environment["torch_cuda"] = torch.version.cuda
            environment["accelerator_available"] = torch.cuda.is_available()
            environment["accelerator_name"] = (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            )
            if not torch.cuda.is_available():
                blockers.append("PyTorch accelerator probe did not pass")
            elif backend == "rocm" and not torch.version.hip:
                blockers.append("PyTorch does not report the configured ROCm backend")
            elif backend == "cuda" and not torch.version.cuda:
                blockers.append("PyTorch does not report the configured CUDA backend")
        except (ImportError, RuntimeError) as exc:
            blockers.append(f"PyTorch ROCm probe failed: {exc}")
    else:
        warnings.append("full ROCm hardware probe was not run")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_file": asdict(config_fingerprint),
        "artifacts": artifacts,
        "code": asdict(code),
        "environment": environment,
        "gpu_budget": budget_manifest,
        "selected_task_count": len(tasks),
    }
    return PreflightReport(
        str(config["experiment_id"]), not blockers, tuple(blockers), tuple(warnings), manifest
    )
