"""Measured GRPO/RLOO stability selection for formal RL arm materialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from .experiment import sha256_file
from .rl_training import validate_adapter_manifest


@dataclass(frozen=True)
class RLStabilityCandidate:
    method: str
    stable_training: bool
    canary_passed: bool
    passed_tasks: int
    total_tasks: int
    strict_success_rate: float
    evidence_unsupported_answer_rate: float | None
    invalid_memory_acceptance_rate: float | None
    mean_tool_calls: float
    max_kl: float
    max_clip_fraction: float
    max_grad_norm: float


@dataclass(frozen=True)
class RLStabilitySelection:
    version: int
    epistemic_status: str
    ready: bool
    selected_method: str | None
    reason: str
    thresholds: dict[str, float]
    candidates: tuple[RLStabilityCandidate, ...]
    sources: dict[str, dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "epistemic_status": self.epistemic_status,
            "ready": self.ready,
            "selected_method": self.selected_method,
            "reason": self.reason,
            "thresholds": self.thresholds,
            "candidates": [asdict(item) for item in self.candidates],
            "sources": self.sources,
        }


def load_stability_thresholds(path: str | Path) -> dict[str, float]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "version", "epistemic_status", "max_kl", "max_clip_fraction", "max_grad_norm"
    } or raw.get("version") != 1:
        raise ValueError("RL stability threshold fields differ from version 1")
    if raw["epistemic_status"] != "frozen_stability_rule_before_gpu_runs":
        raise ValueError("RL stability threshold epistemic status is invalid")
    result = {}
    for field in ("max_kl", "max_clip_fraction", "max_grad_norm"):
        value = raw[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"RL stability {field} must be positive")
        result[field] = float(value)
    if result["max_clip_fraction"] > 1:
        raise ValueError("RL stability max_clip_fraction cannot exceed 1")
    return result


def _fingerprint(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("RL stability source escapes the repository")
    if not resolved.is_file():
        raise ValueError(f"RL stability source is missing: {resolved}")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _training_metrics(
    path: Path, *, method: str, thresholds: dict[str, float], root: Path
) -> tuple[dict[str, float | bool], str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("epistemic_status") != "measured_rl_update_not_evaluation_result":
        raise ValueError(f"{method} training result has invalid epistemic status")
    plan = raw.get("plan")
    history = raw.get("history")
    if not isinstance(plan, dict) or not isinstance(history, list) or not history:
        raise ValueError(f"{method} training result lacks plan/history")
    training = plan.get("training")
    model = plan.get("model")
    if not isinstance(training, dict) or training.get("method") != method:
        raise ValueError(f"{method} training result method drift")
    if not isinstance(model, dict):
        raise ValueError(f"{method} training result lacks model identity")
    model_id = str(model.get("id", ""))
    revision = str(model.get("revision", ""))
    validate_adapter_manifest(
        path, repo_root=root, model_id=model_id, model_revision=revision
    )
    max_steps = training.get("max_steps")
    if not isinstance(max_steps, int) or len(history) != max_steps:
        raise ValueError(f"{method} training history does not cover every configured step")
    required = {"optimizer_step", "loss", "policy_loss", "kl", "clip_fraction", "grad_norm"}
    values: dict[str, list[float]] = {field: [] for field in required}
    for index, row in enumerate(history, 1):
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError(f"{method} training history row is incomplete")
        for field in required:
            value = row[field]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{method} training metric is non-finite: {field}")
            values[field].append(float(value))
        if values["optimizer_step"][-1] != float(index):
            raise ValueError(f"{method} optimizer steps are not contiguous")
    metrics = {
        "max_kl": max(values["kl"]),
        "max_clip_fraction": max(values["clip_fraction"]),
        "max_grad_norm": max(values["grad_norm"]),
    }
    metrics["stable_training"] = all(
        metrics[field] <= thresholds[field]
        for field in ("max_kl", "max_clip_fraction", "max_grad_norm")
    )
    return metrics, model_id, revision


def _canary_metrics(
    manifest_path: Path, *, training_path: Path, model_id: str, revision: str, root: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("stability canary manifest must be version 1")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("stability canary manifest lacks artifacts")
    for name in ("result", "adapter_manifest"):
        entry = artifacts.get(name)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"stability canary lacks artifact {name}")
        path = Path(entry["path"]).resolve()
        if root != path and root not in path.parents:
            raise ValueError("stability canary artifact escapes repository")
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"stability canary artifact hash mismatch: {name}")
        if path.stat().st_size != entry.get("bytes"):
            raise ValueError(f"stability canary artifact size mismatch: {name}")
    adapter_path = Path(artifacts["adapter_manifest"]["path"]).resolve()
    if adapter_path != training_path.resolve():
        raise ValueError("stability canary evaluated another adapter")
    if manifest.get("adapter_manifest_sha256") != sha256_file(training_path):
        raise ValueError("stability canary adapter identity drift")
    if manifest.get("model_id") != model_id or manifest.get("model_revision") != revision:
        raise ValueError("stability canary model identity drift")
    result = json.loads(Path(artifacts["result"]["path"]).read_text(encoding="utf-8"))
    gate = result.get("gate") if isinstance(result, dict) else None
    report = result.get("evaluation_report") if isinstance(result, dict) else None
    conditions = report.get("conditions") if isinstance(report, dict) else None
    if not isinstance(gate, dict) or not isinstance(conditions, list) or len(conditions) != 1:
        raise ValueError("stability canary result lacks gate or one condition")
    summary = conditions[0].get("summary") if isinstance(conditions[0], dict) else None
    if not isinstance(summary, dict):
        raise ValueError("stability canary result lacks summary")
    return {
        "canary_passed": gate.get("passed") is True,
        "passed_tasks": int(gate.get("passed_tasks", -1)),
        "total_tasks": int(gate.get("total_tasks", -1)),
        "strict_success_rate": float(summary["strict_success_rate"]),
        "evidence_unsupported_answer_rate": summary.get("evidence_unsupported_answer_rate"),
        "invalid_memory_acceptance_rate": summary.get("invalid_memory_acceptance_rate"),
        "mean_tool_calls": float(summary["mean_tool_calls"]),
    }


def _rank(candidate: RLStabilityCandidate) -> tuple[float, ...]:
    def lower_is_better(value: float | None) -> float:
        return -float(value) if value is not None else float("-inf")
    return (
        float(candidate.canary_passed), float(candidate.passed_tasks),
        candidate.strict_success_rate,
        lower_is_better(candidate.evidence_unsupported_answer_rate),
        lower_is_better(candidate.invalid_memory_acceptance_rate),
        -candidate.mean_tool_calls, -candidate.max_kl,
        -candidate.max_clip_fraction, -candidate.max_grad_norm,
    )


def select_rl_stability_method(
    *,
    grpo_training_result: str | Path,
    rloo_training_result: str | Path,
    grpo_canary_manifest: str | Path,
    rloo_canary_manifest: str | Path,
    thresholds: dict[str, float],
    repo_root: str | Path,
) -> RLStabilitySelection:
    root = Path(repo_root).resolve()
    sources = {
        "grpo_training": Path(grpo_training_result).resolve(),
        "rloo_training": Path(rloo_training_result).resolve(),
        "grpo_canary": Path(grpo_canary_manifest).resolve(),
        "rloo_canary": Path(rloo_canary_manifest).resolve(),
    }
    candidates = []
    common_identity: tuple[str, str] | None = None
    for method in ("grpo", "rloo"):
        training_path = sources[f"{method}_training"]
        training, model_id, revision = _training_metrics(
            training_path, method=method, thresholds=thresholds, root=root
        )
        if common_identity is None:
            common_identity = (model_id, revision)
        elif common_identity != (model_id, revision):
            raise ValueError("GRPO/RLOO stability candidates use different base models")
        canary = _canary_metrics(
            sources[f"{method}_canary"], training_path=training_path,
            model_id=model_id, revision=revision, root=root,
        )
        candidates.append(RLStabilityCandidate(method=method, **training, **canary))
    eligible = [item for item in candidates if item.stable_training and item.canary_passed]
    selected: str | None = None
    reason = "no candidate passed both frozen training and behavior gates"
    if len(eligible) == 1:
        selected = eligible[0].method
        reason = "only one candidate passed both frozen training and behavior gates"
    elif len(eligible) == 2:
        ranks = {_rank(item): item.method for item in eligible}
        if len(ranks) == 1:
            reason = "both candidates are tied under the frozen ranking; selection remains open"
        else:
            selected = max(eligible, key=_rank).method
            reason = "both passed; selected by frozen behavior-first then stability ranking"
    return RLStabilitySelection(
        version=1,
        epistemic_status="measured_draft_stability_selection_not_formal_evaluation",
        ready=selected is not None, selected_method=selected, reason=reason,
        thresholds=dict(thresholds), candidates=tuple(candidates),
        sources={name: _fingerprint(path, root) for name, path in sources.items()},
    )


def validate_rl_stability_selection(
    raw: object, *, repo_root: str | Path
) -> RLStabilitySelection:
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("RL stability selection must be version 1")
    sources = raw.get("sources")
    thresholds = raw.get("thresholds")
    if not isinstance(sources, dict) or not isinstance(thresholds, dict):
        raise ValueError("RL stability selection lacks sources or thresholds")
    root = Path(repo_root).resolve()
    resolved = {}
    for name in ("grpo_training", "rloo_training", "grpo_canary", "rloo_canary"):
        entry = sources.get(name)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"RL stability selection lacks source {name}")
        path = (root / entry["path"]).resolve()
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"RL stability selection source hash mismatch: {name}")
        if path.stat().st_size != entry.get("bytes"):
            raise ValueError(f"RL stability selection source size mismatch: {name}")
        resolved[name] = path
    rebuilt = select_rl_stability_method(
        grpo_training_result=resolved["grpo_training"],
        rloo_training_result=resolved["rloo_training"],
        grpo_canary_manifest=resolved["grpo_canary"],
        rloo_canary_manifest=resolved["rloo_canary"],
        thresholds={str(key): float(value) for key, value in thresholds.items()},
        repo_root=root,
    )
    if rebuilt.to_json() != raw:
        raise ValueError("RL stability selection report drift")
    return rebuilt
