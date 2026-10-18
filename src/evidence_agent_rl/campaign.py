"""Read-only audit of the staged 100-hour AMD experiment campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .experiment import build_preflight_report, load_experiment_config, sha256_file
from .gpu_budget import load_gpu_budget_ledger, reservation_blocker
from .policy_canary import load_policy_canary_spec
from .formal_rl import load_formal_rl_arm_spec
from .formal_evaluation import load_formal_evaluation_spec


REQUIRED_CAMPAIGN_PHASES = (
    "amd_environment_probe",
    "double_review_gate",
    "sft_smoke",
    "policy_canary",
    "sft_train",
    "base_react_evaluation",
    "sft_evaluation",
    "grpo_stability",
    "rloo_stability",
    "grpo_stability_canary",
    "rloo_stability_canary",
    "rl_no_memory_train",
    "rl_raw_replay_train",
    "rl_query_guidance_train",
    "rl_evidence_bound_train",
    "rl_four_arm_evaluation",
)


@dataclass(frozen=True)
class CampaignPhaseAudit:
    phase_id: str
    kind: str
    max_gpu_hours: float
    status: str
    blockers: tuple[str, ...]
    artifact: str | None
    artifact_sha256: str | None

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["blockers"] = list(self.blockers)
        return result


@dataclass(frozen=True)
class CampaignAudit:
    version: int
    campaign_id: str
    ready: bool
    total_gpu_hours: float
    planned_gpu_hours: float
    contingency_gpu_hours: float
    unallocated_gpu_hours: float
    blockers: tuple[str, ...]
    phases: tuple[CampaignPhaseAudit, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "campaign_id": self.campaign_id,
            "ready": self.ready,
            "total_gpu_hours": self.total_gpu_hours,
            "planned_gpu_hours": self.planned_gpu_hours,
            "contingency_gpu_hours": self.contingency_gpu_hours,
            "unallocated_gpu_hours": self.unallocated_gpu_hours,
            "blockers": list(self.blockers),
            "phases": [phase.to_json() for phase in self.phases],
        }


def load_campaign_spec(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "version", "campaign_id", "epistemic_status", "total_gpu_hours",
        "contingency_gpu_hours", "phases",
    }
    if not isinstance(raw, dict) or set(raw) != required or raw.get("version") != 1:
        raise ValueError("campaign spec fields differ from version 1")
    if raw["epistemic_status"] != "planned_gpu_campaign_no_runs_started":
        raise ValueError("campaign epistemic status is invalid")
    for field in ("total_gpu_hours", "contingency_gpu_hours"):
        value = raw[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"campaign {field} must be non-negative")
    if raw["total_gpu_hours"] != 100:
        raise ValueError("the reserved campaign total must remain 100 GPU hours")
    phases = raw["phases"]
    ids = tuple(item.get("id") for item in phases if isinstance(item, dict)) if isinstance(phases, list) else ()
    if ids != REQUIRED_CAMPAIGN_PHASES:
        raise ValueError("campaign phases must use the frozen order")
    fields = {
        "id", "kind", "experiment_id", "max_gpu_hours", "depends_on", "artifact"
    }
    seen: set[str] = set()
    for phase in phases:
        if not isinstance(phase, dict) or set(phase) != fields:
            raise ValueError("campaign phase fields are invalid")
        if phase["kind"] not in {
            "external_gate", "experiment_config", "policy_canary",
            "deferred_rl_arm", "deferred_evaluation_arm", "deferred_evaluation_family",
        }:
            raise ValueError("campaign phase kind is invalid")
        hours = phase["max_gpu_hours"]
        if not isinstance(hours, (int, float)) or isinstance(hours, bool) or hours < 0:
            raise ValueError("campaign phase hours must be non-negative")
        dependencies = phase["depends_on"]
        if not isinstance(dependencies, list) or any(item not in seen for item in dependencies):
            raise ValueError("campaign dependencies must reference earlier phases")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("campaign phase dependencies must be unique")
        artifact = phase["artifact"]
        experiment_id = phase["experiment_id"]
        if not isinstance(artifact, str) or not artifact:
            raise ValueError("every campaign phase requires an auditable artifact")
        elif phase["kind"] in {
            "experiment_config", "policy_canary", "deferred_rl_arm", "deferred_evaluation_arm"
        } and (
            not isinstance(experiment_id, str) or not experiment_id
        ):
            raise ValueError("GPU phase requires an experiment_id")
        elif phase["kind"] in {"external_gate", "deferred_evaluation_family"} and experiment_id is not None:
            raise ValueError("external gates and combined evaluation families cannot claim one experiment_id")
        if phase["kind"] == "external_gate" and hours != 0:
            raise ValueError("external gates cannot consume repository GPU hours")
        seen.add(phase["id"])
    planned = sum(float(phase["max_gpu_hours"]) for phase in phases)
    if planned + float(raw["contingency_gpu_hours"]) > float(raw["total_gpu_hours"]) + 1e-9:
        raise ValueError("campaign planned plus contingency hours exceed 100")
    return raw


def audit_campaign(spec: dict[str, Any], *, repo_root: str | Path) -> CampaignAudit:
    root = Path(repo_root).resolve()
    phase_results: list[CampaignPhaseAudit] = []
    all_blockers: list[str] = []
    phase_status: dict[str, str] = {}
    ledger_path = root / "configs" / "gpu_budget_ledger.json"
    for phase in spec["phases"]:
        blockers: list[str] = []
        artifact_value = phase["artifact"]
        artifact_path: Path | None = None
        digest: str | None = None
        if artifact_value is None:
            blockers.append("config family awaits stability-method selection and upstream manifests")
        else:
            artifact_path = (root / artifact_value).resolve()
            if root != artifact_path and root not in artifact_path.parents:
                blockers.append("artifact path escapes the repository")
            elif not artifact_path.is_file():
                blockers.append("required artifact is missing")
            else:
                digest = sha256_file(artifact_path)
                try:
                    if phase["kind"] == "experiment_config":
                        config = load_experiment_config(artifact_path)
                        if config["experiment_id"] != phase["experiment_id"]:
                            blockers.append("experiment_id differs from campaign phase")
                        if float(config["runtime"]["max_gpu_hours"]) != float(phase["max_gpu_hours"]):
                            blockers.append("config GPU hours differ from campaign phase")
                        preflight = build_preflight_report(
                            artifact_path,
                            repo_root=root,
                            budget_ledger_path=ledger_path,
                        )
                        blockers.extend(
                            f"preflight: {item}" for item in preflight.blockers
                        )
                    elif phase["kind"] == "policy_canary":
                        canary = load_policy_canary_spec(artifact_path)
                        if canary.experiment_id != phase["experiment_id"]:
                            blockers.append("canary experiment_id differs from campaign phase")
                        if float(canary.max_gpu_hours) != float(phase["max_gpu_hours"]):
                            blockers.append("canary GPU hours differ from campaign phase")
                        ledger_issue = reservation_blocker(
                            load_gpu_budget_ledger(ledger_path),
                            experiment_id=canary.experiment_id,
                            config_sha256=sha256_file(artifact_path),
                            requested_hours=canary.max_gpu_hours,
                        )
                        if ledger_issue:
                            blockers.append(f"preflight: {ledger_issue}")
                    elif phase["kind"] == "external_gate":
                        evidence = json.loads(artifact_path.read_text(encoding="utf-8"))
                        if not isinstance(evidence, dict):
                            blockers.append("external gate evidence must be an object")
                        elif phase["id"] == "amd_environment_probe" and evidence.get("ready") is not True:
                            blockers.append("AMD environment probe is not ready")
                        elif phase["id"] == "double_review_gate" and evidence.get("ready") is not True:
                            blockers.append("double-review gate is not ready")
                    elif phase["kind"] == "deferred_rl_arm":
                        arm_spec = load_formal_rl_arm_spec(artifact_path)
                        arm = arm_spec["arms"].get(phase["id"])
                        if not isinstance(arm, dict):
                            blockers.append("formal RL arm is absent from frozen spec")
                        else:
                            if arm["experiment_id"] != phase["experiment_id"]:
                                blockers.append("formal RL experiment_id differs from campaign phase")
                            if float(arm["max_gpu_hours"]) != float(phase["max_gpu_hours"]):
                                blockers.append("formal RL GPU hours differ from campaign phase")
                        blockers.append(
                            "formal config awaits ready ROCm report, SFT adapter, and measured stability selection"
                        )
                    elif phase["kind"] == "deferred_evaluation_arm":
                        evaluation_spec = load_formal_evaluation_spec(artifact_path)
                        evaluation_arm_id = {
                            "base_react_evaluation": "react_no_rl_no_memory",
                            "sft_evaluation": "sft_only",
                        }.get(phase["id"])
                        if evaluation_arm_id is None:
                            raise ValueError("unknown deferred evaluation campaign phase")
                        arm = evaluation_spec["arms"][evaluation_arm_id]
                        if arm["experiment_id"] != phase["experiment_id"]:
                            blockers.append("formal evaluation identity differs from campaign phase")
                        if float(evaluation_spec["evaluation"]["max_gpu_hours"]) != float(phase["max_gpu_hours"]):
                            blockers.append("formal evaluation hours differ from campaign phase")
                        blockers.append(
                            "formal evaluation config awaits double review and ready ROCm report"
                            + ("" if arm["checkpoint_family"] == "base" else ", plus its adapter")
                        )
                    elif phase["kind"] == "deferred_evaluation_family":
                        evaluation_spec = load_formal_evaluation_spec(artifact_path)
                        rl_arms = [
                            arm for arm in evaluation_spec["arms"].values()
                            if arm["checkpoint_family"] == "rl"
                        ]
                        family_hours = len(rl_arms) * float(
                            evaluation_spec["evaluation"]["max_gpu_hours"]
                        )
                        if len(rl_arms) != 4 or family_hours != float(phase["max_gpu_hours"]):
                            blockers.append("formal RL evaluation family differs from campaign allocation")
                        blockers.append(
                            "formal RL evaluations await double review, ready ROCm report, and four RL adapters"
                        )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    blockers.append(f"artifact validation failed: {exc}")
        for dependency in phase["depends_on"]:
            if phase_status.get(dependency) != "ready":
                blockers.append(f"dependency is not ready: {dependency}")
        status = "ready" if not blockers else "blocked"
        phase_status[phase["id"]] = status
        all_blockers.extend(f"{phase['id']}: {item}" for item in blockers)
        phase_results.append(
            CampaignPhaseAudit(
                phase_id=phase["id"], kind=phase["kind"],
                max_gpu_hours=float(phase["max_gpu_hours"]), status=status,
                blockers=tuple(blockers), artifact=(str(artifact_path) if artifact_path else None),
                artifact_sha256=digest,
            )
        )
    planned = sum(item.max_gpu_hours for item in phase_results)
    total = float(spec["total_gpu_hours"])
    contingency = float(spec["contingency_gpu_hours"])
    return CampaignAudit(
        version=1, campaign_id=spec["campaign_id"], ready=not all_blockers,
        total_gpu_hours=total, planned_gpu_hours=planned,
        contingency_gpu_hours=contingency,
        unallocated_gpu_hours=total - planned - contingency,
        blockers=tuple(all_blockers), phases=tuple(phase_results),
    )
