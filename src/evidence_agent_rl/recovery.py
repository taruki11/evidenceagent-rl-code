"""Draft recovery supervision from failed free-running train trajectories."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .evaluation import EvaluationMatrixReport
from .harness import EvidenceHarness
from .models import (
    Action,
    Document,
    GuidanceKind,
    MemoryMode,
    StrategyMemory,
    TaskSpec,
)
from .policy_skill import PolicySkillContract, coerce_policy_skill
from .react import POLICY_SYSTEM_PROMPT, build_policy_context, serialize_policy_context
from .sft import SFTExample, validate_public_context, validate_sft_examples
from .verifier import DeterministicVerifier


@dataclass(frozen=True)
class RecoveryCandidate:
    candidate_id: str
    source_experiment_id: str
    source_episode_id: str
    source_task_id: str
    source_paper_ids: tuple[str, ...]
    condition_key: str
    failure_category: str
    prefix_action_count: int
    review_status: str
    examples: tuple[SFTExample, ...]

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_paper_ids"] = list(self.source_paper_ids)
        result["examples"] = [example.to_json() for example in self.examples]
        return result


def build_recovery_review_template(
    candidates: Iterable[RecoveryCandidate],
) -> tuple[dict[str, Any], ...]:
    """Create one neutral review row per candidate without pre-approving it."""

    return tuple(
        {
            "reviewer_id": "REPLACE_ME",
            "reviewer_kind": "ai_model",
            "candidate_id": candidate.candidate_id,
            "verdict": "REPLACE_ME",
            "action_sequence_correct": None,
            "memory_boundary_correct": None,
            "evidence_chain_correct": None,
            "answer_correct": None,
            "train_split_safe": None,
            "approved_example_ids": [],
            "notes": "",
        }
        for candidate in candidates
    )


def render_recovery_review_packet(candidates: Iterable[RecoveryCandidate]) -> str:
    """Render a compact audit packet; full public contexts remain in candidate JSONL."""

    rows = tuple(candidates)
    lines = [
        "# EvidenceAgent-RL Recovery SFT 候选审核包",
        "",
        "这些候选只能来自 train split。请逐条检查动作序列、记忆/证据隔离、",
        "当前回合证据链、最终答案以及 source paper split；不要因为轨迹流畅而通过。",
        "两位审核者必须独立填写各自的 JSONL 模板。",
        "",
    ]
    for candidate in rows:
        actions = [
            json.loads(example.messages[-1]["content"])["name"]
            for example in candidate.examples
        ]
        lines.extend(
            [
                f"## {candidate.candidate_id}",
                "",
                f"- source task: `{candidate.source_task_id}`",
                f"- source papers: `{', '.join(candidate.source_paper_ids)}`",
                f"- condition: `{candidate.condition_key}`",
                f"- failure category: `{candidate.failure_category}`",
                f"- preserved prefix actions: `{candidate.prefix_action_count}`",
                f"- proposed actions: `{' -> '.join(actions)}`",
                f"- example IDs: `{', '.join(example.example_id for example in candidate.examples)}`",
                "",
            ]
        )
    return "\n".join(lines)


def _desired_memory_action(
    harness: EvidenceHarness,
    task: TaskSpec,
    memories: dict[str, StrategyMemory],
) -> Action | None:
    if harness.memory_mode is MemoryMode.NONE or not task.available_memory_ids:
        return None
    memory_id = task.available_memory_ids[0]
    if memory_id not in harness.state.inspected_memory_ids:
        return Action("inspect_memory", {"memory_id": memory_id})
    if harness.memory_mode is not MemoryMode.EVIDENCE_BOUND_RECONSTRUCTION:
        return None
    memory = memories.get(memory_id)
    if memory is None:
        return None
    desired_kind = (
        GuidanceKind.EMPTY if task.mismatch_role == "mismatched" else memory.guidance_kind
    )
    desired_instruction = "" if desired_kind is GuidanceKind.EMPTY else memory.instruction
    latest = next(
        (
            guidance
            for guidance in reversed(harness.state.guidance_history)
            if guidance.memory_id == memory_id
        ),
        None,
    )
    if latest is None or (latest.kind, latest.instruction) != (
        desired_kind,
        desired_instruction,
    ):
        return Action(
            "reconstruct_memory",
            {
                "memory_id": memory_id,
                "kind": desired_kind.value,
                "instruction": desired_instruction,
            },
        )
    return None


def _next_recovery_action(
    harness: EvidenceHarness,
    task: TaskSpec,
    memories: dict[str, StrategyMemory],
) -> Action:
    memory_action = _desired_memory_action(harness, task, memories)
    if memory_action is not None:
        return memory_action

    selected = [
        document
        for document in harness.documents.values()
        if document.doi in task.allowed_dois
    ]
    if not harness.state.query_history:
        return Action("search", {"query": task.question})

    if harness.policy_context_version in {"v4", "v5", "v6"}:
        relevant = [
            handle
            for handle in harness.state.evidence.values()
            if handle.checked and set(handle.facets) & set(task.required_facets)
        ]
        if task.task_type == "multi_hop" and len(relevant) == 1:
            bridge_query = f"{task.question} {relevant[0].text}"
            if (
                bridge_query not in harness.state.query_history
                and harness.state.remaining_budget > 0
            ):
                return Action("search", {"query": bridge_query})
        existing_span_ids = {
            handle.span_id for handle in harness.state.evidence.values()
        }
        for document in selected:
            for span in document.spans:
                if (
                    set(span.facets) & set(task.required_facets)
                    and span.span_id not in existing_span_ids
                    and (document.document_id, span.span_id)
                    in harness.public_span_candidates
                ):
                    return Action(
                        "select_evidence",
                        {"document_id": document.document_id, "span_id": span.span_id},
                    )
        if task.insufficient:
            return Action(
                "abstain",
                {"reason": "Current corpus evidence is insufficient for the requested claim."},
            )
        if not harness.state.unresolved_facets and relevant:
            return Action(
                "answer",
                {
                    "claims": [
                        {"text": handle.text, "evidence_handles": [handle.handle_id]}
                        for handle in relevant
                    ]
                },
            )
        if harness.state.remaining_budget > 0:
            return Action("search", {"query": task.question})
        return Action(
            "abstain",
            {"reason": "The remaining claim is not supported by selected evidence."},
        )

    for document in selected:
        target_query = " ".join(document.title for document in selected) or task.question
        if document.document_id not in harness.state.candidate_pool:
            if harness.state.query_history[-1] != target_query:
                return Action("revise_query", {"query": target_query})
            return Action("search", {"query": target_query})
        if document.document_id not in harness.state.retrieved_documents:
            return Action("retrieve", {"document_id": document.document_id})
        existing_span_ids = {
            handle.span_id for handle in harness.state.evidence.values()
        }
        for span in document.spans:
            if (
                set(span.facets) & set(task.required_facets)
                and span.span_id not in existing_span_ids
            ):
                return Action(
                    "inspect_span",
                    {"document_id": document.document_id, "span_id": span.span_id},
                )

    for handle in harness.state.evidence.values():
        if not handle.checked and set(handle.facets) & set(task.required_facets):
            return Action("check_citation", {"handle_id": handle.handle_id})

    if task.insufficient:
        return Action(
            "abstain",
            {"reason": "Current corpus evidence is insufficient for the requested claim."},
        )

    relevant = [
        handle
        for handle in harness.state.evidence.values()
        if handle.checked and set(handle.facets) & set(task.required_facets)
    ]
    if not harness.state.unresolved_facets and relevant:
        return Action(
            "answer",
            {
                "claims": [
                    {"text": handle.text, "evidence_handles": [handle.handle_id]}
                    for handle in relevant
                ]
            },
        )
    return Action(
        "abstain",
        {"reason": "The remaining claim facets are not supported by checked evidence."},
    )


def _failure_prefix(episode) -> tuple[int, str]:
    trace = episode.state.trace
    for index, decision in enumerate(episode.decisions):
        if decision.get("decode_error") is not None:
            return index, "invalid_json_action"
        if index < len(trace) and not trace[index].observation.ok:
            return index, trace[index].observation.kind
    for index, step in enumerate(trace[: len(episode.decisions)]):
        if step.action.name == "answer":
            return index, "premature_or_incorrect_answer"
        if step.action.name == "abstain":
            return index, "premature_or_incorrect_abstention"
    return len(episode.decisions), "decision_limit_or_incomplete_trajectory"


def build_recovery_candidates(
    report: EvaluationMatrixReport,
    documents: Iterable[Document],
    memories: Iterable[StrategyMemory],
    tasks: Iterable[TaskSpec],
    *,
    instructions: str | PolicySkillContract,
    condition_key: str,
    max_recovery_steps: int = 8,
    max_total_decisions: int | None = None,
    policy_context_version: str = "v2",
) -> tuple[RecoveryCandidate, ...]:
    """Build auditable draft corrections; never promote them to reviewed SFT."""

    if report.split != "train":
        raise ValueError("recovery candidates may only be built from train trajectories")
    if max_recovery_steps <= 0:
        raise ValueError("max_recovery_steps must be positive")
    if max_total_decisions is None:
        max_total_decisions = max_recovery_steps
    if max_total_decisions <= 0:
        raise ValueError("max_total_decisions must be positive")
    if policy_context_version not in {"v1", "v2", "v3", "v4", "v5", "v6"}:
        raise ValueError("unsupported recovery policy context version")
    condition = next(
        (item for item in report.conditions if item.condition.key == condition_key), None
    )
    if condition is None:
        raise ValueError("recovery condition is absent from the report")
    document_list = tuple(documents)
    memory_list = tuple(memories)
    memory_map = {memory.memory_id: memory for memory in memory_list}
    task_map = {task.task_id: task for task in tasks}
    contract = coerce_policy_skill(instructions)
    verifier = DeterministicVerifier()
    candidates: list[RecoveryCandidate] = []

    for episode in condition.episodes:
        if episode.strict_success:
            continue
        task = task_map.get(episode.task_id)
        if task is None or task.split != "train":
            raise ValueError("recovery report references an unknown non-train task")
        trajectory_id = f"recovery::{report.experiment_id}::{task.task_id}"
        failure_prefix_count, category = _failure_prefix(episode)
        selected_prefix_count: int | None = None
        selected_examples: tuple[SFTExample, ...] | None = None

        # Preserve as much of the sampled trajectory as possible, but rewind a
        # decision-consuming prefix when the oracle cannot finish from that state.
        # This keeps recovery supervision faithful without silently discarding
        # otherwise recoverable failures at the decision-limit boundary.
        for prefix_count in range(failure_prefix_count, -1, -1):
            recovery_step_cap = min(
                max_recovery_steps,
                max_total_decisions - prefix_count,
            )
            if recovery_step_cap <= 0:
                continue
            harness = EvidenceHarness(
                document_list,
                memory_list,
                max_budget=condition.condition.budget_cap,
                memory_mode=condition.condition.memory_mode,
                policy_context_version=policy_context_version,
            )
            recovery_episode_id = f"recovery:{report.experiment_id}:{task.task_id}"
            harness.reset(task, episode_id=recovery_episode_id)
            for step in episode.state.trace[:prefix_count]:
                harness.step(Action(step.action.name, dict(step.action.arguments)))
            if harness.state.done:
                continue

            examples: list[SFTExample] = []
            for recovery_step in range(recovery_step_cap):
                context = build_policy_context(harness, contract)
                validate_public_context(context)
                action = _next_recovery_action(harness, task, memory_map)
                action_json = json.dumps(
                    {"name": action.name, "arguments": action.arguments},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                examples.append(
                    SFTExample(
                        example_id=f"{trajectory_id}:step-{recovery_step:02d}",
                        task_id=trajectory_id,
                        split="train",
                        source_paper_ids=task.all_source_paper_ids,
                        review_status="draft",
                        step_index=recovery_step,
                        messages=(
                            {"role": "system", "content": POLICY_SYSTEM_PROMPT},
                            {"role": "user", "content": serialize_policy_context(context)},
                            {"role": "assistant", "content": action_json},
                        ),
                    )
                )
                harness.step(action)
                if harness.state.done:
                    break
            if harness.state.done and verifier.is_success(verifier.score(harness.state)):
                selected_prefix_count = prefix_count
                selected_examples = tuple(examples)
                break

        if selected_prefix_count is None or selected_examples is None:
            continue
        validate_sft_examples(selected_examples)
        candidates.append(
            RecoveryCandidate(
                candidate_id=trajectory_id,
                source_experiment_id=report.experiment_id,
                source_episode_id=episode.state.episode_id,
                source_task_id=task.task_id,
                source_paper_ids=task.all_source_paper_ids,
                condition_key=condition_key,
                failure_category=category,
                prefix_action_count=selected_prefix_count,
                review_status="draft_requires_independent_review",
                examples=selected_examples,
            )
        )
    return tuple(candidates)
