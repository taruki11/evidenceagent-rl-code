"""Load and bind the one policy-facing Harness skill contract."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


POLICY_SKILL_NAME = "evidence-research"
REQUIRED_POLICY_ACTIONS = (
    "search",
    "retrieve",
    "inspect_span",
    "check_citation",
    "inspect_memory",
    "reconstruct_memory",
    "revise_query",
    "answer",
    "abstain",
)


@dataclass(frozen=True)
class PolicySkillContract:
    name: str
    description: str
    content: str
    sha256: str

    def identity(self) -> dict[str, str]:
        return {"name": self.name, "sha256": self.sha256}


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _frontmatter(content: str) -> dict[str, str]:
    if not content.startswith("---\n"):
        raise ValueError("policy skill requires YAML frontmatter")
    end = content.find("\n---\n", 4)
    if end < 0:
        raise ValueError("policy skill frontmatter is not closed")
    values: dict[str, str] = {}
    for line in content[4:end].splitlines():
        if ":" not in line:
            raise ValueError("policy skill frontmatter must contain key:value lines")
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    if set(values) != {"name", "description"}:
        raise ValueError("policy skill frontmatter must contain exactly name and description")
    if not all(values.values()):
        raise ValueError("policy skill name and description cannot be empty")
    return values


def load_policy_skill_contract(
    path: str | Path,
    *,
    expected_name: str = POLICY_SKILL_NAME,
    required_actions: tuple[str, ...] = REQUIRED_POLICY_ACTIONS,
) -> PolicySkillContract:
    source = Path(path)
    content = source.read_text(encoding="utf-8")
    metadata = _frontmatter(content)
    if metadata["name"] != expected_name:
        raise ValueError(
            f"policy-facing skill must be {expected_name}, got {metadata['name']}"
        )
    mentioned = set(re.findall(r"`([a-z_]+)(?:\([^`]*\))?`", content))
    missing = set(required_actions) - mentioned
    if missing:
        raise ValueError(f"policy skill omits Harness actions: {sorted(missing)}")
    for boundary in ("## Evidence firewall", "## Stopping rule"):
        if boundary not in content:
            raise ValueError(f"policy skill omits required boundary: {boundary}")
    return PolicySkillContract(
        metadata["name"], metadata["description"], content, _sha256_text(content)
    )


def adapter_policy_skill_sha256(path: str | Path) -> str | None:
    """Read the policy-skill identity frozen into an SFT or RL adapter manifest."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    plan = raw.get("plan") if isinstance(raw, dict) else None
    if not isinstance(plan, dict):
        return None
    dataset = plan.get("dataset")
    if isinstance(dataset, dict):
        digest = dataset.get("policy_skill_sha256")
    else:
        rollout = plan.get("rollout_manifest")
        identity = rollout.get("policy_identity") if isinstance(rollout, dict) else None
        digest = identity.get("policy_skill_sha256") if isinstance(identity, dict) else None
    if digest is None:
        return None
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ValueError("adapter policy skill identity must be lowercase SHA-256")
    return digest


def inline_policy_skill(content: str) -> PolicySkillContract:
    """Compatibility helper for unit-level providers, never a formal run artifact."""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("policy instructions cannot be empty")
    return PolicySkillContract("inline-unversioned", "Inline test contract", content, _sha256_text(content))


def coerce_policy_skill(value: str | PolicySkillContract) -> PolicySkillContract:
    if isinstance(value, PolicySkillContract):
        return value
    return inline_policy_skill(value)


def validate_policy_skill_binding(context: dict[str, Any]) -> None:
    instructions = context.get("instructions")
    identity = context.get("skill_contract")
    if not isinstance(instructions, str) or not instructions:
        raise ValueError("policy context lacks skill instructions")
    if not isinstance(identity, dict) or set(identity) != {"name", "sha256"}:
        raise ValueError("policy context lacks a closed skill_contract identity")
    if not isinstance(identity["name"], str) or not identity["name"]:
        raise ValueError("policy skill name is invalid")
    if identity["sha256"] != _sha256_text(instructions):
        raise ValueError("policy skill content does not match its SHA-256 identity")
