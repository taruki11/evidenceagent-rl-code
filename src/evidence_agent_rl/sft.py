"""Leakage-checked action-supervision export from successful harness trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from .harness import EvidenceHarness
from .models import Action, Document, EpisodeState, MemoryMode, StrategyMemory, TaskSpec
from .policies import (
    OracleEvidenceMemoryPolicy,
    PublicExhaustionTeacher,
    PublicMultiHopTeacher,
)
from .react import POLICY_SYSTEM_PROMPT, build_policy_context, serialize_policy_context
from .policy_skill import PolicySkillContract, validate_policy_skill_binding
from .verifier import DeterministicVerifier


FORBIDDEN_POLICY_KEYS = frozenset(
    {
        "required_facets",
        "deterministic_answer_key",
        "forbidden_answer_tokens",
        "required_query_terms",
        "allowed_dois",
        "mismatch_role",
        "mismatch_pair_id",
        "coverage_graph",
        "unresolved_facets",
    }
)


@dataclass(frozen=True)
class SFTExample:
    example_id: str
    task_id: str
    split: str
    source_paper_ids: tuple[str, ...]
    review_status: str
    step_index: int
    messages: tuple[dict[str, str], ...]

    def to_json(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "task_id": self.task_id,
            "split": self.split,
            "source_paper_ids": list(self.source_paper_ids),
            "review_status": self.review_status,
            "step_index": self.step_index,
            "messages": list(self.messages),
        }


def _find_forbidden_keys(value: object, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in FORBIDDEN_POLICY_KEYS:
                found.append(child_path)
            found.extend(_find_forbidden_keys(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_keys(child, f"{path}[{index}]"))
    return found


def validate_public_context(context: dict[str, object]) -> None:
    forbidden = _find_forbidden_keys(context)
    if forbidden:
        raise ValueError(f"policy context leaks hidden gold fields: {forbidden}")


def validate_sft_examples(examples: Iterable[SFTExample]) -> None:
    seen_examples: set[str] = set()
    paper_splits: dict[str, str] = {}
    count = 0
    for example in examples:
        count += 1
        if example.example_id in seen_examples:
            raise ValueError(f"duplicate SFT example_id: {example.example_id}")
        seen_examples.add(example.example_id)
        if tuple(message.get("role") for message in example.messages) != (
            "system", "user", "assistant"
        ):
            raise ValueError(f"{example.example_id}: invalid message roles")
        context = json.loads(example.messages[1]["content"])
        validate_public_context(context)
        validate_policy_skill_binding(context)
        action = json.loads(example.messages[2]["content"])
        if action.get("name") == "abstain":
            reason = str(action.get("arguments", {}).get("reason", "")).casefold()
            if "oracle" in reason or "gold" in reason:
                raise ValueError(
                    f"{example.example_id}: abstention label leaks hidden teacher state"
                )
            if context.get("policy_context_version") in {"v3", "v4", "v5", "v6"}:
                policy_state = context.get("policy_state", {})
                retrieved_span_ids = {
                    str(span_id)
                    for document in policy_state.get("retrieved_document_ledger", [])
                    for span_id in document.get("span_ids", [])
                }
                inspected_span_ids = {
                    str(evidence.get("span_id"))
                    for evidence in policy_state.get("evidence_ledger", [])
                }
            else:
                retrieved_span_ids = {
                    str(span_id)
                    for step in context.get("trace", [])
                    if step.get("observation", {}).get("kind") == "document"
                    for span_id in step.get("observation", {}).get("data", {}).get("span_ids", [])
                }
                inspected_span_ids = {
                    str(step.get("action", {}).get("arguments", {}).get("span_id"))
                    for step in context.get("trace", [])
                    if step.get("action", {}).get("name") == "inspect_span"
                }
            remaining = context.get("policy_state", {}).get("remaining_budget")
            if (
                isinstance(remaining, int)
                and remaining > 0
                and retrieved_span_ids - inspected_span_ids
            ):
                raise ValueError(
                    f"{example.example_id}: abstention label is premature in public state"
                )
        for source_paper_id in example.source_paper_ids:
            previous = paper_splits.setdefault(source_paper_id, example.split)
            if previous != example.split:
                raise ValueError(
                    f"source paper {source_paper_id} leaks across {previous} and {example.split}"
                )
    if count == 0:
        raise ValueError("SFT dataset must contain at least one example")


def validate_public_multihop_answer_binding(state: EpisodeState) -> None:
    """Require the public teacher to bind each hop to its own checked evidence."""

    if state.task.task_type != "multi_hop":
        raise ValueError("public multi-hop answer binding requires a multi_hop task")
    if not state.done or len(state.final_claims) < 2:
        raise ValueError("public multi-hop answer requires at least two atomic claims")
    if any(len(claim.evidence_handles) != 1 for claim in state.final_claims):
        raise ValueError("public multi-hop answer claims must each bind one evidence handle")
    checked_handles = {
        handle_id
        for handle_id, checked in state.verification_records.items()
        if checked
    }
    cited_handles = [
        handle_id
        for claim in state.final_claims
        for handle_id in claim.evidence_handles
    ]
    if set(cited_handles) != checked_handles:
        raise ValueError(
            "public multi-hop answer must cite every checked evidence handle exactly once"
        )
    if len(cited_handles) != len(set(cited_handles)):
        raise ValueError(
            "public multi-hop answer must cite every checked evidence handle exactly once"
        )


def build_sft_examples(
    tasks: Iterable[TaskSpec],
    documents: Iterable[Document],
    memories: Iterable[StrategyMemory],
    *,
    instructions: str | PolicySkillContract,
    max_budget: int = 8,
    policy_context_version: str = "v2",
) -> tuple[SFTExample, ...]:
    task_list = tuple(tasks)
    document_list = tuple(documents)
    memory_list = tuple(memories)
    oracle_teacher = OracleEvidenceMemoryPolicy()
    public_exhaustion_teacher = PublicExhaustionTeacher()
    public_multihop_teacher = PublicMultiHopTeacher()
    verifier = DeterministicVerifier()
    examples: list[SFTExample] = []
    for task in task_list:
        episode_id = f"sft:{task.task_id}"
        rollout = EvidenceHarness(
            document_list,
            memory_list,
            max_budget=max_budget,
            memory_mode=(
                MemoryMode.NONE
                if policy_context_version in {"v4", "v5", "v6"}
                else MemoryMode.EVIDENCE_BOUND_RECONSTRUCTION
            ),
            policy_context_version=policy_context_version,
        )
        if task.insufficient:
            teacher = public_exhaustion_teacher
        elif task.task_type == "multi_hop":
            teacher = public_multihop_teacher
        else:
            teacher = oracle_teacher
        teacher.run(rollout, task, episode_id=episode_id)
        if task.task_type == "multi_hop":
            validate_public_multihop_answer_binding(rollout.state)
        score = verifier.score(rollout.state)
        if not verifier.is_success(score):
            raise ValueError(
                f"teacher trajectory failed for {task.task_id}: reward={score.total}"
            )
        prefix = EvidenceHarness(
            document_list,
            memory_list,
            max_budget=max_budget,
            memory_mode=(
                MemoryMode.NONE
                if policy_context_version in {"v4", "v5", "v6"}
                else MemoryMode.EVIDENCE_BOUND_RECONSTRUCTION
            ),
            policy_context_version=policy_context_version,
        )
        prefix.reset(task, episode_id=episode_id)
        for step_index, step in enumerate(rollout.state.trace):
            context = build_policy_context(prefix, instructions)
            validate_public_context(context)
            action_json = json.dumps(
                {"name": step.action.name, "arguments": step.action.arguments},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            examples.append(
                SFTExample(
                    example_id=f"{task.task_id}:step-{step_index:02d}",
                    task_id=task.task_id,
                    split=task.split,
                    source_paper_ids=task.all_source_paper_ids,
                    review_status=task.review_status,
                    step_index=step_index,
                    messages=(
                        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": serialize_policy_context(context),
                        },
                        {"role": "assistant", "content": action_json},
                    ),
                )
            )
            prefix.step(Action(step.action.name, dict(step.action.arguments)))
        if prefix.state != rollout.state:
            raise AssertionError(f"prefix replay diverged for {task.task_id}")
    result = tuple(examples)
    validate_sft_examples(result)
    return result


def audit_public_multihop_tasks(
    tasks: Iterable[TaskSpec],
    documents: Iterable[Document],
    memories: Iterable[StrategyMemory],
    *,
    max_budget: int = 8,
    policy_context_version: str = "v2",
) -> dict[str, object]:
    """Measure which multi-hop tasks a public-observation teacher can replay."""

    task_list = tuple(tasks)
    document_list = tuple(documents)
    memory_list = tuple(memories)
    teacher = PublicMultiHopTeacher()
    verifier = DeterministicVerifier()
    cases: list[dict[str, object]] = []
    for task in task_list:
        if task.task_type != "multi_hop":
            continue
        harness = EvidenceHarness(
            document_list,
            memory_list,
            max_budget=max_budget,
            memory_mode=(
                MemoryMode.NONE
                if policy_context_version in {"v4", "v5", "v6"}
                else MemoryMode.EVIDENCE_BOUND_RECONSTRUCTION
            ),
            policy_context_version=policy_context_version,
        )
        error: str | None = None
        try:
            teacher.run(harness, task, episode_id=f"public-audit:{task.task_id}")
            validate_public_multihop_answer_binding(harness.state)
            score = verifier.score(harness.state)
            strict_success = verifier.is_success(score)
        except (RuntimeError, TypeError, ValueError) as exc:
            strict_success = False
            error = str(exc)
        cases.append(
            {
                "task_id": task.task_id,
                "strict_success": strict_success,
                "actions": [step.action.name for step in harness.state.trace],
                "remaining_budget": harness.state.remaining_budget,
                "error": error,
            }
        )
    eligible_ids = [str(case["task_id"]) for case in cases if case["strict_success"]]
    excluded_ids = [str(case["task_id"]) for case in cases if not case["strict_success"]]
    return {
        "version": 1,
        "epistemic_status": "public_observation_multihop_achievability_not_model_performance",
        "max_budget": max_budget,
        "policy_context_version": policy_context_version,
        "multi_hop_tasks": len(cases),
        "eligible_multi_hop_tasks": len(eligible_ids),
        "excluded_multi_hop_tasks": len(excluded_ids),
        "eligible_task_ids": eligible_ids,
        "excluded_task_ids": excluded_ids,
        "cases": cases,
    }
