"""Frozen, cross-report comparison matrix for EvidenceAgent-RL baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .experiment import sha256_file


REQUIRED_ARM_IDS = (
    "fixed_rag",
    "react_no_rl_no_memory",
    "sft_only",
    "rl_search_no_memory",
    "rl_raw_replay",
    "rl_query_guidance",
    "rl_evidence_bound_reconstruction",
    "oracle_evidence",
)
MEMORY_MODES = frozenset(
    {"none", "raw_replay", "query_guidance", "evidence_bound_reconstruction"}
)
EXPECTED_ARM_CONTRACTS = {
    "fixed_rag": ("deterministic", "fixed_rag", None, None, "none"),
    "react_no_rl_no_memory": ("evaluation", "react", "base", "none", "none"),
    "sft_only": (
        "evaluation", "react", "sft", "evidence_bound_reconstruction",
        "evidence_bound_reconstruction",
    ),
    "rl_search_no_memory": ("evaluation", "react", "rl", "none", "none"),
    "rl_raw_replay": ("evaluation", "react", "rl", "raw_replay", "raw_replay"),
    "rl_query_guidance": (
        "evaluation", "react", "rl", "query_guidance", "query_guidance",
    ),
    "rl_evidence_bound_reconstruction": (
        "evaluation", "react", "rl", "evidence_bound_reconstruction",
        "evidence_bound_reconstruction",
    ),
    "oracle_evidence": ("deterministic", "oracle_evidence", None, None, "none"),
}


@dataclass(frozen=True)
class ComparisonArmResult:
    arm_id: str
    label: str
    status: str
    blockers: tuple[str, ...]
    checkpoint_kind: str | None
    training_memory_mode: str | None
    inference_memory_mode: str
    manifest_path: str
    manifest_sha256: str | None
    model_id: str | None
    model_revision: str | None
    task_ids: tuple[str, ...]
    data_fingerprints: dict[str, str]
    metrics: dict[str, Any] | None

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["blockers"] = list(self.blockers)
        result["task_ids"] = list(self.task_ids)
        return result


@dataclass(frozen=True)
class ComparisonMatrixPlan:
    version: int
    matrix_id: str
    epistemic_status: str
    ready: bool
    blockers: tuple[str, ...]
    arms: tuple[ComparisonArmResult, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "matrix_id": self.matrix_id,
            "epistemic_status": self.epistemic_status,
            "ready": self.ready,
            "blockers": list(self.blockers),
            "arms": [arm.to_json() for arm in self.arms],
        }


def load_comparison_matrix_spec(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "version", "matrix_id", "epistemic_status", "comparison_contract",
        "model", "arms",
    }
    if not isinstance(raw, dict) or set(raw) != required or raw.get("version") != 1:
        raise ValueError("comparison matrix spec fields differ from version 1")
    if raw.get("epistemic_status") not in {
        "planned_formal_comparison_no_results", "human_reviewed_formal_comparison"
    }:
        raise ValueError("comparison matrix epistemic status is invalid")
    contract = raw["comparison_contract"]
    if not isinstance(contract, dict) or set(contract) != {
        "split", "quality_gate", "budget_cap", "required_task_order",
    }:
        raise ValueError("comparison contract fields are invalid")
    if contract["split"] not in {"test", "ood"}:
        raise ValueError("formal comparison split must be test or ood")
    if contract["quality_gate"] != "double_review":
        raise ValueError("formal comparison requires double_review")
    if not isinstance(contract["budget_cap"], int) or contract["budget_cap"] <= 0:
        raise ValueError("comparison budget_cap must be positive")
    if contract["required_task_order"] != "exact":
        raise ValueError("comparison requires exact task ordering")
    model = raw["model"]
    if not isinstance(model, dict) or set(model) != {"id", "revision"}:
        raise ValueError("comparison model identity is invalid")
    if not all(isinstance(model[key], str) and model[key] for key in model):
        raise ValueError("comparison model identity must be non-empty")
    arms = raw["arms"]
    ids = tuple(arm.get("id") for arm in arms if isinstance(arm, dict)) if isinstance(arms, list) else ()
    if ids != REQUIRED_ARM_IDS:
        raise ValueError("comparison arms must use the frozen eight-arm order")
    arm_fields = {
        "id", "label", "source_type", "policy", "checkpoint_kind",
        "training_memory_mode", "inference_memory_mode", "manifest",
        "manifest_sha256",
    }
    for arm in arms:
        if not isinstance(arm, dict) or set(arm) != arm_fields:
            raise ValueError("comparison arm fields are invalid")
        if arm["source_type"] not in {"deterministic", "evaluation"}:
            raise ValueError("comparison arm source_type is invalid")
        if arm["inference_memory_mode"] not in MEMORY_MODES:
            raise ValueError("comparison arm inference memory mode is invalid")
        if arm["source_type"] == "deterministic":
            if arm["checkpoint_kind"] is not None or arm["training_memory_mode"] is not None:
                raise ValueError("deterministic arms cannot declare a checkpoint")
            if arm["policy"] not in {"fixed_rag", "oracle_evidence"}:
                raise ValueError("deterministic arm policy is invalid")
            if arm["inference_memory_mode"] != "none":
                raise ValueError("deterministic arms must use memory mode none")
        else:
            if arm["checkpoint_kind"] not in {"base", "sft", "rl"}:
                raise ValueError("model arm checkpoint kind is invalid")
            if arm["training_memory_mode"] not in MEMORY_MODES:
                raise ValueError("model arm training memory mode is invalid")
            if arm["policy"] != "react":
                raise ValueError("model arms must execute ReAct through Harness")
        if not isinstance(arm["manifest"], str) or not arm["manifest"]:
            raise ValueError("comparison arm manifest path is required")
        if not isinstance(arm["manifest_sha256"], str) or not arm["manifest_sha256"]:
            raise ValueError("comparison arm manifest_sha256 is required")
        observed_contract = (
            arm["source_type"], arm["policy"], arm["checkpoint_kind"],
            arm["training_memory_mode"], arm["inference_memory_mode"],
        )
        if observed_contract != EXPECTED_ARM_CONTRACTS[arm["id"]]:
            raise ValueError(f"frozen comparison semantics changed: {arm['id']}")
    return raw


def materialize_comparison_matrix_spec(
    template_path: str | Path, *, repo_root: str | Path
) -> dict[str, Any]:
    """Freeze all eight manifest hashes only when the full matrix validates."""

    root = Path(repo_root).resolve()
    template_file = Path(template_path).resolve()
    if root != template_file and root not in template_file.parents:
        raise ValueError("comparison matrix template must live inside the repository")
    spec = load_comparison_matrix_spec(template_file)
    if spec["epistemic_status"] != "planned_formal_comparison_no_results":
        raise ValueError("comparison matrix materializer requires the planned template")
    result = json.loads(json.dumps(spec))
    for arm in result["arms"]:
        manifest = (root / arm["manifest"]).resolve()
        if root != manifest and root not in manifest.parents:
            raise ValueError(f"comparison manifest escapes repository: {arm['id']}")
        if not manifest.is_file():
            raise ValueError(f"comparison manifest is missing: {arm['id']}")
        arm["manifest_sha256"] = sha256_file(manifest)
    result["epistemic_status"] = "human_reviewed_formal_comparison"
    plan = build_comparison_matrix_plan(result, repo_root=root)
    if not plan.ready:
        raise ValueError("formal comparison is not ready: " + "; ".join(plan.blockers))
    return result


def _artifact_hashes(manifest: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return {}, ["manifest lacks artifacts"]
    blockers: list[str] = []
    hashes: dict[str, str] = {}
    for name in ("documents", "tasks", "memories", "pages", "result"):
        entry = artifacts.get(name)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            blockers.append(f"manifest lacks artifact {name}")
            continue
        path = Path(entry["path"])
        if not path.is_file():
            blockers.append(f"artifact is missing: {name}")
            continue
        digest = sha256_file(path)
        if digest != entry.get("sha256") or path.stat().st_size != entry.get("bytes"):
            blockers.append(f"artifact fingerprint mismatch: {name}")
        elif name != "result":
            hashes[name] = digest
    return hashes, blockers


def _checkpoint_blockers(
    arm: dict[str, Any], manifest: dict[str, Any], model: dict[str, str]
) -> list[str]:
    if arm["source_type"] == "deterministic":
        return []
    blockers: list[str] = []
    if manifest.get("model_id") != model["id"] or manifest.get("model_revision") != model["revision"]:
        blockers.append("model identity differs from the frozen matrix")
    adapter_entry = manifest.get("artifacts", {}).get("adapter_manifest")
    if arm["checkpoint_kind"] == "base":
        if adapter_entry is not None:
            blockers.append("base ReAct arm must not load an adapter")
        return blockers
    if not isinstance(adapter_entry, dict) or not isinstance(adapter_entry.get("path"), str):
        return blockers + ["model arm lacks adapter_manifest"]
    path = Path(adapter_entry["path"])
    if not path.is_file() or sha256_file(path) != adapter_entry.get("sha256"):
        return blockers + ["adapter manifest fingerprint mismatch"]
    try:
        adapter = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return blockers + ["adapter manifest is not readable JSON"]
    if not isinstance(adapter, dict):
        return blockers + ["adapter manifest must be an object"]
    expected_status = (
        "measured_training_run_not_evaluation_result"
        if arm["checkpoint_kind"] == "sft"
        else "measured_rl_update_not_evaluation_result"
    )
    if adapter.get("epistemic_status") != expected_status:
        blockers.append("adapter checkpoint kind differs from the frozen arm")
    if arm["checkpoint_kind"] == "rl":
        observed = adapter.get("plan", {}).get("rollout_manifest", {}).get("memory_mode")
        if observed != arm["training_memory_mode"]:
            blockers.append("RL training memory mode differs from the frozen arm")
    return blockers


def build_comparison_matrix_plan(
    spec: dict[str, Any], *, repo_root: str | Path
) -> ComparisonMatrixPlan:
    root = Path(repo_root).resolve()
    contract = spec["comparison_contract"]
    arm_results: list[ComparisonArmResult] = []
    all_blockers: list[str] = []
    reference_tasks: tuple[str, ...] | None = None
    reference_hashes: dict[str, str] | None = None
    for arm in spec["arms"]:
        blockers: list[str] = []
        path = (root / arm["manifest"]).resolve()
        path_allowed = root == path or root in path.parents
        if not path_allowed:
            blockers.append("manifest path escapes the repository")
        expected_hash = arm["manifest_sha256"]
        if expected_hash.startswith("REPLACE_WITH_"):
            blockers.append("manifest hash is not frozen")
        manifest: dict[str, Any] = {}
        report: dict[str, Any] = {}
        hashes: dict[str, str] = {}
        task_ids: tuple[str, ...] = ()
        metrics: dict[str, Any] | None = None
        observed_hash: str | None = None
        model_id: str | None = None
        model_revision: str | None = None
        if not path_allowed:
            pass
        elif not path.is_file():
            blockers.append("manifest is missing")
        else:
            observed_hash = sha256_file(path)
            if observed_hash != expected_hash:
                blockers.append("manifest hash differs from the frozen arm")
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict) or manifest.get("version") != 1:
                    raise ValueError("manifest must be a version 1 object")
                hashes, issues = _artifact_hashes(manifest)
                blockers.extend(issues)
                if not issues:
                    result_path = Path(manifest["artifacts"]["result"]["path"])
                    report = json.loads(result_path.read_text(encoding="utf-8"))
                    if not isinstance(report, dict):
                        raise ValueError("comparison source report must be an object")
            except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
                blockers.append(f"manifest/report load failed: {exc}")
        if report:
            raw_ids = report.get("task_ids")
            if isinstance(raw_ids, list) and all(isinstance(item, str) for item in raw_ids):
                task_ids = tuple(raw_ids)
            else:
                blockers.append("report task_ids are invalid")
            if report.get("split") != contract["split"] or manifest.get("split") != contract["split"]:
                blockers.append("split differs from the comparison contract")
            if arm["source_type"] == "deterministic":
                if report.get("policy") != arm["policy"]:
                    blockers.append("deterministic policy differs from the frozen arm")
                if report.get("max_budget") != contract["budget_cap"]:
                    blockers.append("deterministic budget differs from the comparison contract")
                if report.get("epistemic_status") != "human_reviewed_deterministic_baseline":
                    blockers.append("deterministic report is not human reviewed")
                if manifest.get("minimum_review") != contract["quality_gate"]:
                    blockers.append("deterministic review gate differs from the comparison contract")
                metrics = report.get("summary") if isinstance(report.get("summary"), dict) else None
            else:
                if manifest.get("quality_gate") != contract["quality_gate"]:
                    blockers.append("evaluation review gate differs from the comparison contract")
                if report.get("epistemic_status") != "human_reviewed_evaluation":
                    blockers.append("model report is not human reviewed evaluation")
                model_id = report.get("model_id")
                model_revision = report.get("model_revision")
                blockers.extend(_checkpoint_blockers(arm, manifest, spec["model"]))
                condition_key = f"memory={arm['inference_memory_mode']}|budget={contract['budget_cap']}"
                matches = [
                    item for item in report.get("conditions", [])
                    if isinstance(item, dict) and item.get("condition_key") == condition_key
                ]
                if len(matches) != 1:
                    blockers.append("required evaluation condition is absent or duplicated")
                elif not isinstance(matches[0].get("summary"), dict):
                    blockers.append("evaluation condition summary is invalid")
                else:
                    metrics = matches[0]["summary"]
        if task_ids and not blockers:
            if reference_tasks is None:
                reference_tasks, reference_hashes = task_ids, hashes
            else:
                if task_ids != reference_tasks:
                    blockers.append("task IDs/order differ from another ready arm")
                if hashes != reference_hashes:
                    blockers.append("data artifact hashes differ from another ready arm")
        status = "ready" if not blockers else "blocked"
        all_blockers.extend(f"{arm['id']}: {issue}" for issue in blockers)
        arm_results.append(
            ComparisonArmResult(
                arm_id=arm["id"], label=arm["label"], status=status,
                blockers=tuple(blockers), checkpoint_kind=arm["checkpoint_kind"],
                training_memory_mode=arm["training_memory_mode"],
                inference_memory_mode=arm["inference_memory_mode"],
                manifest_path=str(path), manifest_sha256=observed_hash,
                model_id=model_id, model_revision=model_revision,
                task_ids=task_ids, data_fingerprints=hashes, metrics=metrics,
            )
        )
    return ComparisonMatrixPlan(
        version=1, matrix_id=spec["matrix_id"],
        epistemic_status=spec["epistemic_status"], ready=not all_blockers,
        blockers=tuple(all_blockers), arms=tuple(arm_results),
    )
