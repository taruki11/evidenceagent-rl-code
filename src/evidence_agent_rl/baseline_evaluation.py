"""Replay-audited deterministic baselines over the common EvidenceHarness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .harness import EvidenceHarness
from .io import dump_trajectory
from .metrics import EvaluationSummary, summarize_episodes
from .models import Document, MemoryMode, StrategyMemory, TaskSpec
from .policies import FixedRAGPolicy, OracleEvidencePolicy
from .verifier import DeterministicVerifier, RewardBreakdown


BASELINE_POLICIES = frozenset({"fixed_rag", "oracle_evidence"})


def load_deterministic_baseline_config(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "version", "experiment_id", "epistemic_status", "policies", "data",
        "split", "minimum_review", "max_budget", "fixed_rag_top_k",
    }
    if not isinstance(raw, dict) or set(raw) != required or raw.get("version") != 1:
        raise ValueError("deterministic baseline config fields differ from version 1")
    if raw["epistemic_status"] not in {
        "draft_deterministic_baseline_not_model_result",
        "human_reviewed_deterministic_baseline",
    }:
        raise ValueError("invalid deterministic baseline config epistemic status")
    if raw["split"] not in {"train", "dev", "test", "ood"}:
        raise ValueError("invalid deterministic baseline split")
    if raw["minimum_review"] not in {"draft", "single_review", "double_review"}:
        raise ValueError("invalid deterministic baseline review gate")
    if raw["epistemic_status"].startswith("human_reviewed") and raw["minimum_review"] == "draft":
        raise ValueError("human-reviewed baseline cannot use the draft gate")
    policies = raw["policies"]
    if not isinstance(policies, list) or set(policies) != BASELINE_POLICIES or len(policies) != 2:
        raise ValueError("baseline config must contain fixed_rag and oracle_evidence once")
    data = raw["data"]
    if not isinstance(data, dict) or set(data) != {"documents", "tasks", "memories", "pages"}:
        raise ValueError("deterministic baseline data paths are invalid")
    if any(not isinstance(value, str) or not value for value in data.values()):
        raise ValueError("deterministic baseline data paths must be strings")
    if not isinstance(raw["max_budget"], int) or not 1 <= raw["max_budget"] <= 12:
        raise ValueError("deterministic baseline max_budget must be in [1, 12]")
    if not isinstance(raw["fixed_rag_top_k"], int) or raw["fixed_rag_top_k"] <= 0:
        raise ValueError("deterministic baseline fixed_rag_top_k must be positive")
    if not isinstance(raw["experiment_id"], str) or not raw["experiment_id"]:
        raise ValueError("deterministic baseline experiment_id is required")
    return raw


@dataclass(frozen=True)
class DeterministicBaselineEpisode:
    task_id: str
    strict_success: bool
    score: RewardBreakdown
    trajectory: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "strict_success": self.strict_success,
            "score": {**asdict(self.score), "total": self.score.total},
            "trajectory": self.trajectory,
        }


@dataclass(frozen=True)
class DeterministicBaselineReport:
    version: int
    epistemic_status: str
    experiment_id: str
    policy: str
    split: str
    max_budget: int
    fixed_rag_top_k: int | None
    task_ids: tuple[str, ...]
    summary: EvaluationSummary
    episodes: tuple[DeterministicBaselineEpisode, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "epistemic_status": self.epistemic_status,
            "experiment_id": self.experiment_id,
            "policy": self.policy,
            "split": self.split,
            "max_budget": self.max_budget,
            "fixed_rag_top_k": self.fixed_rag_top_k,
            "task_ids": list(self.task_ids),
            "summary": asdict(self.summary),
            "episodes": [episode.to_json() for episode in self.episodes],
        }


def run_deterministic_baseline(
    *,
    experiment_id: str,
    policy_name: str,
    documents: Iterable[Document],
    memories: Iterable[StrategyMemory],
    tasks: Iterable[TaskSpec],
    max_budget: int,
    fixed_rag_top_k: int = 1,
    epistemic_status: str,
) -> DeterministicBaselineReport:
    task_list = tuple(tasks)
    document_list = tuple(documents)
    memory_list = tuple(memories)
    if policy_name not in BASELINE_POLICIES:
        raise ValueError(f"unknown deterministic baseline: {policy_name}")
    if not task_list or len({task.task_id for task in task_list}) != len(task_list):
        raise ValueError("deterministic baseline tasks must be unique and non-empty")
    if len({task.split for task in task_list}) != 1:
        raise ValueError("deterministic baseline requires exactly one split")
    if not isinstance(max_budget, int) or not 1 <= max_budget <= 12:
        raise ValueError("deterministic baseline max_budget must be in [1, 12]")
    if not isinstance(fixed_rag_top_k, int) or fixed_rag_top_k <= 0:
        raise ValueError("fixed_rag_top_k must be positive")
    allowed_statuses = {
        "draft_deterministic_baseline_not_model_result",
        "human_reviewed_deterministic_baseline",
    }
    if epistemic_status not in allowed_statuses:
        raise ValueError("invalid deterministic baseline epistemic status")
    policy = (
        FixedRAGPolicy(top_k=fixed_rag_top_k)
        if policy_name == "fixed_rag"
        else OracleEvidencePolicy()
    )
    verifier = DeterministicVerifier()
    states = []
    episodes = []
    for task in task_list:
        harness = EvidenceHarness(
            document_list, memory_list, max_budget=max_budget, memory_mode=MemoryMode.NONE
        )
        policy.run(
            harness,
            task,
            episode_id=f"baseline:{experiment_id}:{policy.name}:{task.task_id}",
        )
        score = verifier.score(harness.state)
        states.append(harness.state)
        episodes.append(
            DeterministicBaselineEpisode(
                task.task_id,
                verifier.is_success(score),
                score,
                dump_trajectory(harness.state),
            )
        )
    return DeterministicBaselineReport(
        version=1,
        epistemic_status=epistemic_status,
        experiment_id=experiment_id,
        policy=policy_name,
        split=task_list[0].split,
        max_budget=max_budget,
        fixed_rag_top_k=fixed_rag_top_k if policy_name == "fixed_rag" else None,
        task_ids=tuple(task.task_id for task in task_list),
        summary=summarize_episodes(states),
        episodes=tuple(episodes),
    )


def validate_deterministic_baseline_report(
    raw: object,
    *,
    documents: Iterable[Document],
    memories: Iterable[StrategyMemory],
    tasks: Iterable[TaskSpec],
) -> DeterministicBaselineReport:
    if not isinstance(raw, dict):
        raise ValueError("deterministic baseline report must be an object")
    required = {
        "version", "epistemic_status", "experiment_id", "policy", "split",
        "max_budget", "fixed_rag_top_k", "task_ids", "summary", "episodes",
    }
    if set(raw) != required or raw.get("version") != 1:
        raise ValueError("deterministic baseline report fields differ from version 1")
    task_map = {task.task_id: task for task in tasks}
    task_ids = raw.get("task_ids")
    if not isinstance(task_ids, list) or not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("deterministic baseline task IDs are invalid")
    if unknown := set(task_ids) - set(task_map):
        raise ValueError(f"deterministic baseline contains unknown tasks: {sorted(unknown)}")
    selected = tuple(task_map[task_id] for task_id in task_ids)
    rebuilt = run_deterministic_baseline(
        experiment_id=str(raw.get("experiment_id", "")),
        policy_name=str(raw.get("policy", "")),
        documents=documents,
        memories=memories,
        tasks=selected,
        max_budget=raw.get("max_budget"),
        fixed_rag_top_k=(raw.get("fixed_rag_top_k") or 1),
        epistemic_status=str(raw.get("epistemic_status", "")),
    )
    if rebuilt.to_json() != raw:
        raise ValueError("deterministic baseline report diverges from policy rerun")
    return rebuilt
