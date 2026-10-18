"""Static completion audit for the CPU-side EvidenceAgent-RL infrastructure."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .campaign import audit_campaign, load_campaign_spec
from .comparison_matrix import load_comparison_matrix_spec
from .experiment import hash_code_tree, sha256_file
from .formal_evaluation import load_formal_evaluation_spec
from .formal_rl import load_formal_rl_arm_spec
from .policy_skill import load_policy_skill_contract
from .stability import load_stability_thresholds


REQUIRED_INFRASTRUCTURE_ARTIFACTS = (
    "src/evidence_agent_rl/harness.py",
    "src/evidence_agent_rl/verifier.py",
    "src/evidence_agent_rl/react.py",
    "src/evidence_agent_rl/sft_training.py",
    "src/evidence_agent_rl/rl_training.py",
    "src/evidence_agent_rl/formal_rl.py",
    "src/evidence_agent_rl/formal_evaluation.py",
    "src/evidence_agent_rl/online_rl.py",
    "skills/evidence-research/SKILL.md",
    "skills/rl-config-materialization/SKILL.md",
    "skills/formal-rl-arms/SKILL.md",
    "skills/formal-evaluation/SKILL.md",
    "configs/amd_campaign_v1.json",
    "configs/formal_rl_arms_v1.json",
    "configs/formal_evaluation_arms_v1.json",
    "configs/baseline_checkpoint_matrix.template.json",
    "configs/rl_stability_thresholds_v1.json",
    "scripts/materialize_sft_config.py",
    "scripts/materialize_rl_config.py",
    "scripts/materialize_formal_rl_arm.py",
    "scripts/materialize_formal_evaluation.py",
    "scripts/materialize_baseline_comparison.py",
    "scripts/audit_evaluation_report.py",
    "scripts/audit_amd_campaign.py",
    "scripts/audit_infrastructure.py",
)


@dataclass(frozen=True)
class InfrastructureAudit:
    version: int
    epistemic_status: str
    infrastructure_ready: bool
    campaign_ready: bool
    blockers: tuple[str, ...]
    external_blockers: tuple[str, ...]
    code_tree_sha256: str
    policy_skill: dict[str, Any]
    artifacts: dict[str, dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["blockers"] = list(self.blockers)
        result["external_blockers"] = list(self.external_blockers)
        return result


def build_infrastructure_audit(*, repo_root: str | Path) -> InfrastructureAudit:
    root = Path(repo_root).resolve()
    blockers: list[str] = []
    artifacts: dict[str, dict[str, Any]] = {}
    for relative in REQUIRED_INFRASTRUCTURE_ARTIFACTS:
        path = (root / relative).resolve()
        if root != path and root not in path.parents:
            blockers.append(f"required artifact escapes repository: {relative}")
        elif not path.is_file():
            blockers.append(f"required artifact is missing: {relative}")
        else:
            artifacts[relative] = {
                "sha256": sha256_file(path), "bytes": path.stat().st_size
            }
    policy_identity: dict[str, Any] = {}
    campaign_ready = False
    external_blockers: list[str] = []
    if not blockers:
        try:
            policy_identity = load_policy_skill_contract(
                root / "skills" / "evidence-research" / "SKILL.md"
            ).identity()
            load_formal_rl_arm_spec(root / "configs" / "formal_rl_arms_v1.json")
            load_formal_evaluation_spec(root / "configs" / "formal_evaluation_arms_v1.json")
            load_comparison_matrix_spec(
                root / "configs" / "baseline_checkpoint_matrix.template.json"
            )
            load_stability_thresholds(root / "configs" / "rl_stability_thresholds_v1.json")
            campaign_spec = load_campaign_spec(root / "configs" / "amd_campaign_v1.json")
            campaign = audit_campaign(campaign_spec, repo_root=root)
            if campaign.planned_gpu_hours != 68.5 or campaign.contingency_gpu_hours != 31.5:
                blockers.append("campaign budget differs from frozen 100-hour allocation")
            campaign_ready = campaign.ready
            external_blockers.extend(campaign.blockers)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"infrastructure contract validation failed: {exc}")
    forbidden = "d:" + "\\pycharm_workplace" + "\\ai_agent"
    for folder in ("src", "scripts", "skills", "configs"):
        for path in (root / folder).rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".md", ".json", ".toml"}:
                try:
                    content = path.read_text(encoding="utf-8").casefold()
                except UnicodeDecodeError:
                    continue
                if forbidden in content:
                    blockers.append(
                        f"forbidden company-code path referenced by infrastructure: {path.relative_to(root)}"
                    )
    ready = not blockers
    return InfrastructureAudit(
        version=1,
        epistemic_status=(
            "cpu_infrastructure_validated_no_gpu_claim"
            if ready else "cpu_infrastructure_incomplete"
        ),
        infrastructure_ready=ready,
        campaign_ready=campaign_ready,
        blockers=tuple(blockers),
        external_blockers=tuple(external_blockers),
        code_tree_sha256=hash_code_tree(root),
        policy_skill=policy_identity,
        artifacts=artifacts,
    )
