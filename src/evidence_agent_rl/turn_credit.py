"""State-anchored action credit for multi-turn Harness rollouts."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from typing import Any, Iterable

from .models import Document, TaskSpec
from .rl import compute_group_advantages
from .trainer_bridge import trainer_batch_from_json
from .verifier import _matches_token


TERMINAL_ACTIONS = {"answer", "abstain"}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def classify_state_gain(steps: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Label observable state gain without consulting hidden task gold."""

    seen_actions: set[str] = set()
    seen_search_results: set[str] = set()
    seen_retrieved_documents: set[str] = set()
    seen_evidence: set[str] = set()
    seen_citations: set[str] = set()
    seen_other: set[str] = set()
    result: list[dict[str, Any]] = []
    for step in steps:
        action = step.get("action") or {}
        name = str(action.get("name", ""))
        observation = step.get("observation") or {}
        data = observation.get("data") or {}
        action_key = _canonical(action)
        repeated = action_key in seen_actions
        seen_actions.add(action_key)
        terminal = name in TERMINAL_ACTIONS
        invalid = observation.get("ok") is not True or bool(observation.get("error"))
        productive = False
        if not invalid and not terminal and not repeated:
            kind = observation.get("kind")
            if kind == "search_results":
                current = {str(item) for item in data.get("document_ids", [])}
                productive = bool(current - seen_search_results)
                seen_search_results.update(current)
            elif kind == "document":
                document_id = str(data.get("document_id", ""))
                productive = bool(
                    document_id and document_id not in seen_retrieved_documents
                )
                seen_retrieved_documents.add(document_id)
            elif kind == "evidence":
                handle_id = str(data.get("handle_id", ""))
                productive = bool(handle_id and handle_id not in seen_evidence)
                seen_evidence.add(handle_id)
            elif kind == "citation_check":
                handle_id = str(data.get("handle_id", ""))
                productive = bool(data.get("checked") and handle_id not in seen_citations)
                if data.get("checked"):
                    seen_citations.add(handle_id)
            else:
                observation_key = _canonical(observation)
                productive = observation_key not in seen_other
                seen_other.add(observation_key)
        result.append(
            {
                "step_index": int(step.get("index", len(result))),
                "action": name,
                "terminal": terminal,
                "invalid": invalid,
                "repeated": repeated,
                "productive": productive,
            }
        )
    return result


def _anchor_key(sample: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(sample["messages"]).encode("utf-8")).hexdigest()


def _replace_rollout_ids(value: object, rollout_ids: tuple[str, ...]) -> object:
    """Remove episode-local identities while preserving the public state."""

    if isinstance(value, str):
        normalized = value
        for rollout_id in rollout_ids:
            normalized = normalized.replace(rollout_id, "<ROLLOUT_ID>")
        return normalized
    if isinstance(value, list):
        return [_replace_rollout_ids(item, rollout_ids) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_rollout_ids(item, rollout_ids) for item in value)
    if isinstance(value, dict):
        return {
            key: _replace_rollout_ids(item, rollout_ids)
            for key, item in value.items()
        }
    return value


def _anchor_key_v2(sample: dict[str, Any], rollout_ids: tuple[str, ...]) -> str:
    public_messages = _replace_rollout_ids(sample["messages"], rollout_ids)
    return hashlib.sha256(_canonical(public_messages).encode("utf-8")).hexdigest()


def _lexicographic_utilities(samples: list[dict[str, Any]]) -> list[float]:
    """Rank terminal correctness first and composite reward only as a tie-break."""

    outcomes: list[tuple[bool, float]] = []
    for sample in samples:
        breakdown = sample.get("reward_breakdown") or {}
        composite = float(breakdown.get("total", sample["trajectory_reward"]))
        outcomes.append((bool(sample.get("strict_success")), composite))
    ordered = {value: rank for rank, value in enumerate(sorted(set(outcomes)))}
    return [float(ordered[value]) for value in outcomes]


def relabel_state_anchored_credit(
    raw: object, *, experiment_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare actions only at identical public states and guard local credit signs.

    Exact-state anchors with multiple actions and reward variance receive GRPO
    advantages. Productive intermediate actions cannot receive negative credit;
    invalid or repeated actions cannot receive positive credit. Unmatched states
    receive zero credit rather than being compared across different states.
    """

    if not isinstance(raw, dict) or not experiment_id.strip():
        raise ValueError(
            "state-anchored credit requires a rollout payload and experiment_id"
        )
    batch = trainer_batch_from_json(raw.get("trainer_batch"))
    rollouts = raw.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) < 2:
        raise ValueError("state-anchored credit requires at least two rollouts")
    rollout_ids = {str(row.get("rollout_id", "")) for row in rollouts}
    if "" in rollout_ids or len(rollout_ids) != len(rollouts):
        raise ValueError("rollout IDs must be non-empty and unique")
    if {sample.rollout_id for sample in batch.samples} != rollout_ids:
        raise ValueError("trainer samples and rollout rows have different rollout IDs")

    state_gain: dict[tuple[str, int], dict[str, Any]] = {}
    for rollout in rollouts:
        rollout_id = str(rollout["rollout_id"])
        steps = (rollout.get("trajectory") or {}).get("steps", [])
        for row in classify_state_gain(steps):
            state_gain[(rollout_id, row["step_index"])] = row

    sample_rows = raw["trainer_batch"]["samples"]
    anchors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in sample_rows:
        key = (str(sample["rollout_id"]), int(sample["step_index"]))
        if key not in state_gain:
            raise ValueError(f"trainer sample has no matching trajectory step: {key}")
        anchors[_anchor_key(sample)].append(sample)

    credit_by_sample = {str(sample["sample_id"]): 0.0 for sample in sample_rows}
    eligible_anchors = 0
    eligible_samples = 0
    raw_negative_productive = 0
    raw_positive_bad = 0
    protected_negative_productive = 0
    suppressed_positive_bad = 0
    for samples in anchors.values():
        rewards = [float(sample["trajectory_reward"]) for sample in samples]
        if (
            len({str(sample["rollout_id"]) for sample in samples}) < 2
            or len({str(sample["action_text"]) for sample in samples}) < 2
            or max(rewards) == min(rewards)
        ):
            continue
        eligible_anchors += 1
        eligible_samples += len(samples)
        raw_advantages = compute_group_advantages(rewards, method="grpo")
        for sample, raw_advantage in zip(samples, raw_advantages):
            label = state_gain[
                (str(sample["rollout_id"]), int(sample["step_index"]))
            ]
            if label["terminal"]:
                credit = raw_advantage
            elif label["productive"]:
                raw_negative_productive += raw_advantage < 0.0
                credit = max(raw_advantage, 0.0)
                protected_negative_productive += raw_advantage < 0.0
            elif label["invalid"] or label["repeated"]:
                raw_positive_bad += raw_advantage > 0.0
                credit = min(raw_advantage, 0.0)
                suppressed_positive_bad += raw_advantage > 0.0
            else:
                raw_positive_bad += raw_advantage > 0.0
                credit = 0.0
                suppressed_positive_bad += raw_advantage > 0.0
            credit_by_sample[str(sample["sample_id"])] = credit

    transformed = deepcopy(raw)
    transformed_batch = transformed["trainer_batch"]
    transformed_batch["experiment_id"] = experiment_id
    transformed_batch["advantage_method"] = "state_anchored_grpo_v1"
    for sample in transformed_batch["samples"]:
        sample["advantage"] = credit_by_sample[str(sample["sample_id"])]
    trainer_batch_from_json(transformed_batch)

    nonzero = sum(abs(value) > 1e-12 for value in credit_by_sample.values())
    report = {
        "version": 1,
        "epistemic_status": "cpu_state_anchored_credit_contract_not_model_training",
        "anchors": len(anchors),
        "eligible_anchors": eligible_anchors,
        "samples": len(sample_rows),
        "eligible_samples": eligible_samples,
        "nonzero_credit_samples": nonzero,
        "raw_negative_productive": raw_negative_productive,
        "raw_positive_invalid_repeated_or_no_gain": raw_positive_bad,
        "protected_negative_productive": protected_negative_productive,
        "suppressed_positive_invalid_repeated_or_no_gain": suppressed_positive_bad,
        "post_guard_negative_productive": 0,
        "post_guard_positive_invalid_repeated_or_no_gain": 0,
    }
    transformed["credit_assignment"] = report
    return transformed, report


def relabel_state_anchored_credit_v2(
    raw: object, *, experiment_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assign signed credit only among actions from the same canonical public state.

    Episode-local rollout IDs are removed from the state anchor.  Action utility
    is ranked lexicographically by strict terminal success and then by the
    existing deterministic composite reward.  Productive actions may therefore
    receive negative credit; invalid or repeated actions can never receive
    positive credit.  Frozen trajectories and recorded rewards are unchanged.
    """

    if not isinstance(raw, dict) or not experiment_id.strip():
        raise ValueError(
            "state-anchored credit requires a rollout payload and experiment_id"
        )
    batch = trainer_batch_from_json(raw.get("trainer_batch"))
    rollouts = raw.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) < 2:
        raise ValueError("state-anchored credit requires at least two rollouts")
    rollout_ids = {str(row.get("rollout_id", "")) for row in rollouts}
    if "" in rollout_ids or len(rollout_ids) != len(rollouts):
        raise ValueError("rollout IDs must be non-empty and unique")
    if {sample.rollout_id for sample in batch.samples} != rollout_ids:
        raise ValueError("trainer samples and rollout rows have different rollout IDs")
    canonical_rollout_ids = tuple(sorted(rollout_ids, key=len, reverse=True))

    state_gain: dict[tuple[str, int], dict[str, Any]] = {}
    for rollout in rollouts:
        rollout_id = str(rollout["rollout_id"])
        steps = (rollout.get("trajectory") or {}).get("steps", [])
        for row in classify_state_gain(steps):
            state_gain[(rollout_id, row["step_index"])] = row

    sample_rows = raw["trainer_batch"]["samples"]
    anchors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in sample_rows:
        key = (str(sample["rollout_id"]), int(sample["step_index"]))
        if key not in state_gain:
            raise ValueError(f"trainer sample has no matching trajectory step: {key}")
        anchors[_anchor_key_v2(sample, canonical_rollout_ids)].append(sample)

    credit_by_sample = {str(sample["sample_id"]): 0.0 for sample in sample_rows}
    eligible_anchors = 0
    eligible_samples = 0
    suppressed_positive_bad = 0
    action_nonzero: dict[str, int] = defaultdict(int)
    terminal_nonzero = 0
    for samples in anchors.values():
        utilities = _lexicographic_utilities(samples)
        if (
            len({str(sample["rollout_id"]) for sample in samples}) < 2
            or len({str(sample["action_text"]) for sample in samples}) < 2
            or max(utilities) == min(utilities)
        ):
            continue
        eligible_anchors += 1
        eligible_samples += len(samples)
        raw_advantages = compute_group_advantages(utilities, method="grpo")
        for sample, raw_advantage in zip(samples, raw_advantages):
            label = state_gain[
                (str(sample["rollout_id"]), int(sample["step_index"]))
            ]
            credit = raw_advantage
            if (label["invalid"] or label["repeated"]) and credit > 0.0:
                credit = 0.0
                suppressed_positive_bad += 1
            credit_by_sample[str(sample["sample_id"])] = credit
            if abs(credit) > 1e-12:
                action_nonzero[str(label["action"])] += 1
                terminal_nonzero += int(label["terminal"])

    transformed = deepcopy(raw)
    transformed_batch = transformed["trainer_batch"]
    transformed_batch["experiment_id"] = experiment_id
    transformed_batch["advantage_method"] = "state_anchored_grpo_v2"
    for sample in transformed_batch["samples"]:
        sample["advantage"] = credit_by_sample[str(sample["sample_id"])]
    trainer_batch_from_json(transformed_batch)

    values = tuple(credit_by_sample.values())
    report = {
        "version": 2,
        "epistemic_status": "cpu_state_anchored_credit_contract_not_model_training",
        "anchor_contract": "canonical_public_messages_without_rollout_ids",
        "utility_contract": "lexicographic_strict_success_then_composite_reward_rank",
        "anchors": len(anchors),
        "eligible_anchors": eligible_anchors,
        "samples": len(sample_rows),
        "eligible_samples": eligible_samples,
        "nonzero_credit_samples": sum(abs(value) > 1e-12 for value in values),
        "positive_credit_samples": sum(value > 1e-12 for value in values),
        "negative_credit_samples": sum(value < -1e-12 for value in values),
        "terminal_nonzero_credit_samples": terminal_nonzero,
        "nonzero_credit_by_action": dict(sorted(action_nonzero.items())),
        "suppressed_positive_invalid_or_repeated": suppressed_positive_bad,
    }
    transformed["credit_assignment"] = report
    return transformed, report


def relabel_state_anchored_turn_credit_v3(
    raw: object, *, experiment_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Combine exact-state comparisons with bounded local transition credit.

    V2 is retained wherever two rollouts expose different actions at the same
    canonical public state.  States after that divergence are not comparable,
    so previously-zero actions receive a small deterministic local label:
    productive public transitions are positive, invalid/repeated/no-gain
    transitions are negative, and terminal actions use strict success.  No
    hidden answer, facet, or citation gold is consulted for intermediate turns.
    """

    transformed, v2_report = relabel_state_anchored_credit_v2(
        raw, experiment_id=experiment_id
    )
    rollouts = transformed["rollouts"]
    state_gain: dict[tuple[str, int], dict[str, Any]] = {}
    for rollout in rollouts:
        rollout_id = str(rollout["rollout_id"])
        steps = (rollout.get("trajectory") or {}).get("steps", [])
        for row in classify_state_gain(steps):
            state_gain[(rollout_id, row["step_index"])] = row

    action_nonzero: dict[str, int] = defaultdict(int)
    local_credit_samples = 0
    terminal_credit_samples = 0
    positive = 0
    negative = 0
    for sample in transformed["trainer_batch"]["samples"]:
        credit = float(sample["advantage"])
        label = state_gain[(str(sample["rollout_id"]), int(sample["step_index"]))]
        if abs(credit) <= 1e-12:
            if label["terminal"]:
                credit = 1.0 if bool(sample.get("strict_success")) else -1.0
                terminal_credit_samples += 1
            elif label["invalid"] or label["repeated"]:
                credit = -0.5
                local_credit_samples += 1
            elif label["productive"]:
                credit = 0.25
                local_credit_samples += 1
            else:
                credit = -0.25
                local_credit_samples += 1
            sample["advantage"] = credit
        if abs(credit) > 1e-12:
            action_nonzero[str(label["action"])] += 1
            positive += credit > 0.0
            negative += credit < 0.0

    transformed["trainer_batch"]["advantage_method"] = "state_anchored_turn_credit_v3"
    trainer_batch_from_json(transformed["trainer_batch"])
    report = {
        "version": 3,
        "epistemic_status": "cpu_state_anchored_dense_turn_credit_not_model_training",
        "anchor_contract": v2_report["anchor_contract"],
        "utility_contract": (
            "v2_exact_state_credit_then_bounded_public_transition_and_terminal_credit"
        ),
        "anchors": v2_report["anchors"],
        "eligible_anchors": v2_report["eligible_anchors"],
        "samples": v2_report["samples"],
        "eligible_samples": v2_report["eligible_samples"],
        "nonzero_credit_samples": positive + negative,
        "positive_credit_samples": positive,
        "negative_credit_samples": negative,
        "terminal_nonzero_credit_samples": sum(
            count
            for action, count in action_nonzero.items()
            if action in TERMINAL_ACTIONS
        ),
        "local_transition_credit_samples": local_credit_samples,
        "terminal_fallback_credit_samples": terminal_credit_samples,
        "nonzero_credit_by_action": dict(sorted(action_nonzero.items())),
        "v2_nonzero_credit_samples": v2_report["nonzero_credit_samples"],
    }
    transformed["credit_assignment"] = report
    return transformed, report


def relabel_verifier_progress_turn_credit_v4(
    raw: object,
    *,
    experiment_id: str,
    task: TaskSpec,
    documents: Iterable[Document],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Label each turn by deterministic query and required-facet progress.

    Gold answer text is never exposed to the policy.  The offline labeler uses
    only the frozen verifier contract: required query terms, required facet
    IDs, allowed DOIs, and strict terminal success.  This distinguishes a
    useful span/citation from merely touching a new span or handle.
    """

    if not isinstance(raw, dict) or not experiment_id.strip():
        raise ValueError("verifier progress credit requires a payload and experiment_id")
    batch = trainer_batch_from_json(raw.get("trainer_batch"))
    if batch.task_id != task.task_id:
        raise ValueError("rollout batch and verifier task IDs differ")
    document_map = {document.document_id: document for document in documents}
    required_facets = set(task.required_facets)
    required_terms = tuple(task.required_query_terms)
    labels: dict[tuple[str, int], dict[str, Any]] = {}
    query_gain_samples = 0
    facet_inspection_gain_samples = 0
    facet_check_gain_samples = 0

    for rollout in raw.get("rollouts", []):
        rollout_id = str(rollout["rollout_id"])
        strict_success = bool(rollout.get("strict_success"))
        covered_terms: set[str] = set()
        inspected_facets: set[str] = set()
        checked_facets: set[str] = set()
        handle_facets: dict[str, set[str]] = {}
        seen_actions: set[str] = set()
        for step in (rollout.get("trajectory") or {}).get("steps", []):
            index = int(step.get("index", 0))
            action = step.get("action") or {}
            name = str(action.get("name", ""))
            arguments = action.get("arguments") or {}
            observation = step.get("observation") or {}
            action_key = _canonical(action)
            repeated = action_key in seen_actions
            seen_actions.add(action_key)
            invalid = observation.get("ok") is not True or bool(observation.get("error"))
            credit = -1.0 if invalid else -0.5 if repeated else -0.25
            gain = "none"

            if not invalid and not repeated and name in {"search", "revise_query"}:
                query = str(arguments.get("query", ""))
                new_terms = {
                    term for term in required_terms if _matches_token(term, query)
                } - covered_terms
                covered_terms.update(new_terms)
                if new_terms:
                    credit = min(1.0, 0.5 * len(new_terms))
                    query_gain_samples += 1
                    gain = "required_query_term"
                elif not required_terms:
                    returned = (observation.get("data") or {}).get("document_ids", [])
                    useful = any(
                        document_map.get(str(document_id)) is not None
                        and document_map[str(document_id)].doi in task.allowed_dois
                        for document_id in returned
                    )
                    credit = 0.25 if useful else -0.25
                    gain = "allowed_document" if useful else "none"
            elif not invalid and not repeated and name == "retrieve":
                document = document_map.get(str(arguments.get("document_id", "")))
                useful = document is not None and document.doi in task.allowed_dois
                credit = 0.25 if useful else -0.5
                gain = "allowed_document" if useful else "none"
            elif not invalid and not repeated and name == "inspect_span":
                document = document_map.get(str(arguments.get("document_id", "")))
                span_id = str(arguments.get("span_id", ""))
                span = next(
                    (item for item in document.spans if item.span_id == span_id),
                    None,
                ) if document is not None else None
                facets = set(span.facets) & required_facets if span is not None else set()
                new_facets = facets - inspected_facets
                inspected_facets.update(new_facets)
                handle_id = str((observation.get("data") or {}).get("handle_id", ""))
                if handle_id:
                    handle_facets[handle_id] = facets
                if new_facets:
                    credit = 0.75
                    facet_inspection_gain_samples += 1
                    gain = "required_facet_inspected"
            elif not invalid and not repeated and name == "check_citation":
                facets = handle_facets.get(str(arguments.get("handle_id", "")), set())
                new_facets = facets - checked_facets
                checked_facets.update(new_facets)
                if new_facets:
                    credit = 1.0
                    facet_check_gain_samples += 1
                    gain = "required_facet_checked"
            elif name in TERMINAL_ACTIONS:
                credit = 1.0 if strict_success else -1.0
                gain = "strict_terminal_success" if strict_success else "terminal_failure"
            elif not invalid and not repeated:
                credit = 0.1
                gain = "valid_control_action"

            labels[(rollout_id, index)] = {
                "action": name,
                "credit": credit,
                "gain": gain,
                "invalid": invalid,
                "repeated": repeated,
            }

    transformed = deepcopy(raw)
    action_nonzero: dict[str, int] = defaultdict(int)
    gain_counts: dict[str, int] = defaultdict(int)
    positive = 0
    negative = 0
    invalid_or_repeated_positive = 0
    for sample in transformed["trainer_batch"]["samples"]:
        key = (str(sample["rollout_id"]), int(sample["step_index"]))
        if key not in labels:
            raise ValueError(f"trainer sample has no verifier progress label: {key}")
        label = labels[key]
        sample["advantage"] = label["credit"]
        action_nonzero[label["action"]] += 1
        gain_counts[label["gain"]] += 1
        positive += label["credit"] > 0.0
        negative += label["credit"] < 0.0
        invalid_or_repeated_positive += (
            (label["invalid"] or label["repeated"]) and label["credit"] > 0.0
        )
    transformed["trainer_batch"]["experiment_id"] = experiment_id
    transformed["trainer_batch"]["advantage_method"] = "verifier_progress_turn_credit_v4"
    trainer_batch_from_json(transformed["trainer_batch"])
    report = {
        "version": 4,
        "epistemic_status": "cpu_verifier_progress_turn_credit_not_model_training",
        "task_type": str(task.task_type),
        "samples": len(transformed["trainer_batch"]["samples"]),
        "eligible_anchors": 0,
        "eligible_samples": len(transformed["trainer_batch"]["samples"]),
        "nonzero_credit_samples": positive + negative,
        "positive_credit_samples": positive,
        "negative_credit_samples": negative,
        "invalid_or_repeated_positive_process_samples": invalid_or_repeated_positive,
        "terminal_nonzero_credit_samples": sum(
            count for action, count in action_nonzero.items() if action in TERMINAL_ACTIONS
        ),
        "query_gain_samples": query_gain_samples,
        "facet_inspection_gain_samples": facet_inspection_gain_samples,
        "facet_check_gain_samples": facet_check_gain_samples,
        "nonzero_credit_by_action": dict(sorted(action_nonzero.items())),
        "gain_counts": dict(sorted(gain_counts.items())),
        "utility_contract": "frozen_required_query_and_facet_progress_with_strict_terminal_credit",
    }
    transformed["credit_assignment"] = report
    return transformed, report


def relabel_hybrid_outcome_verifier_progress_v5(
    raw: object,
    *,
    experiment_id: str,
    task: TaskSpec,
    documents: Iterable[Document],
    process_weight: float = 0.25,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep trajectory outcome as the update direction and use progress as shaping."""

    if not 0.0 < process_weight <= 0.25:
        raise ValueError("hybrid verifier progress weight must be in (0, 0.25]")
    transformed, process_report = relabel_verifier_progress_turn_credit_v4(
        raw,
        experiment_id=experiment_id,
        task=task,
        documents=documents,
    )
    rollouts = tuple(raw.get("rollouts", [])) if isinstance(raw, dict) else ()
    outcomes = tuple(bool(rollout.get("strict_success")) for rollout in rollouts)
    if set(outcomes) != {False, True}:
        raise ValueError("hybrid outcome credit requires a mixed success/failure group")
    rollout_ids = tuple(str(rollout["rollout_id"]) for rollout in rollouts)
    success_by_rollout = dict(zip(rollout_ids, outcomes))
    outcome_advantages = dict(
        zip(
            rollout_ids,
            compute_group_advantages(
                tuple(1.0 if success else 0.0 for success in outcomes),
                method="grpo",
            ),
        )
    )

    success_credits: list[float] = []
    failure_credits: list[float] = []
    process_positive = 0
    process_negative = 0
    hybrid_positive = 0
    hybrid_negative = 0
    for sample in transformed["trainer_batch"]["samples"]:
        rollout_id = str(sample["rollout_id"])
        if rollout_id not in outcome_advantages:
            raise ValueError(f"trainer sample has unknown rollout: {rollout_id}")
        process_credit = float(sample["advantage"])
        hybrid_credit = outcome_advantages[rollout_id] + process_weight * process_credit
        sample["advantage"] = hybrid_credit
        process_positive += process_credit > 0.0
        process_negative += process_credit < 0.0
        hybrid_positive += hybrid_credit > 0.0
        hybrid_negative += hybrid_credit < 0.0
        target = success_credits if success_by_rollout[rollout_id] else failure_credits
        target.append(hybrid_credit)

    successful_actions_positive = bool(success_credits) and min(success_credits) > 0.0
    failed_actions_negative = bool(failure_credits) and max(failure_credits) < 0.0
    if not successful_actions_positive or not failed_actions_negative:
        raise ValueError("hybrid credit failed to preserve the trajectory outcome sign")

    method = "hybrid_outcome_verifier_progress_v5"
    transformed["trainer_batch"]["advantage_method"] = method
    trainer_batch_from_json(transformed["trainer_batch"])
    report = {
        **process_report,
        "version": 5,
        "epistemic_status": "cpu_hybrid_outcome_verifier_progress_credit_not_model_training",
        "outcome_weight": 1.0,
        "process_weight": process_weight,
        "mixed_outcome_group": True,
        "successful_actions_positive": successful_actions_positive,
        "failed_actions_negative": failed_actions_negative,
        "minimum_successful_action_credit": min(success_credits),
        "maximum_failed_action_credit": max(failure_credits),
        "process_positive_samples": process_positive,
        "process_negative_samples": process_negative,
        "invalid_or_repeated_positive_process_samples": process_report[
            "invalid_or_repeated_positive_process_samples"
        ],
        "positive_credit_samples": hybrid_positive,
        "negative_credit_samples": hybrid_negative,
        "nonzero_credit_samples": hybrid_positive + hybrid_negative,
        "utility_contract": "binary_outcome_grpo_direction_plus_bounded_verifier_progress_shaping",
    }
    transformed["credit_assignment"] = report
    return transformed, report


def relabel_multihop_graph_progress_v6(
    raw: object,
    *,
    experiment_id: str,
    task: TaskSpec,
    documents: Iterable[Document],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assign potential-difference credit on a frozen multi-hop evidence graph.

    The potential contains only verifier milestones that can be established
    from the recorded environment transition: required query coverage,
    discovery and retrieval of allowed documents, required-facet inspection,
    and citation checking.  No gold answer text is copied into policy inputs.
    Valid zero-progress bridge actions receive zero rather than a blanket
    penalty; invalid and repeated actions remain negative.
    """

    if not isinstance(raw, dict) or not experiment_id.strip():
        raise ValueError("multi-hop graph progress requires a payload and experiment_id")
    task_type = task.task_type.value if hasattr(task.task_type, "value") else str(task.task_type)
    if task_type != "multi_hop":
        raise ValueError("multi-hop graph progress may only label multi_hop tasks")
    batch = trainer_batch_from_json(raw.get("trainer_batch"))
    if batch.task_id != task.task_id:
        raise ValueError("rollout batch and graph-progress task IDs differ")

    document_map = {document.document_id: document for document in documents}
    allowed_document_ids = {
        document.document_id
        for document in document_map.values()
        if document.doi in task.allowed_dois
    }
    required_facets = set(task.required_facets)
    required_terms = set(task.required_query_terms)

    def fraction(items: set[str], required: set[str]) -> float:
        return len(items & required) / len(required) if required else 0.0

    def potential(
        covered_terms: set[str],
        discovered_documents: set[str],
        retrieved_documents: set[str],
        inspected_facets: set[str],
        checked_facets: set[str],
    ) -> float:
        return (
            0.20 * fraction(covered_terms, required_terms)
            + 0.10 * fraction(discovered_documents, allowed_document_ids)
            + 0.15 * fraction(retrieved_documents, allowed_document_ids)
            + 0.25 * fraction(inspected_facets, required_facets)
            + 0.30 * fraction(checked_facets, required_facets)
        )

    labels: dict[tuple[str, int], dict[str, Any]] = {}
    milestone_counts: dict[str, int] = defaultdict(int)
    rollout_final_potentials: list[float] = []
    for rollout in raw.get("rollouts", []):
        rollout_id = str(rollout["rollout_id"])
        strict_success = bool(rollout.get("strict_success"))
        covered_terms: set[str] = set()
        discovered_documents: set[str] = set()
        retrieved_documents: set[str] = set()
        inspected_facets: set[str] = set()
        checked_facets: set[str] = set()
        handle_facets: dict[str, set[str]] = {}
        seen_actions: set[str] = set()

        for step in (rollout.get("trajectory") or {}).get("steps", []):
            index = int(step.get("index", 0))
            action = step.get("action") or {}
            name = str(action.get("name", ""))
            arguments = action.get("arguments") or {}
            observation = step.get("observation") or {}
            action_key = _canonical(action)
            repeated = action_key in seen_actions
            seen_actions.add(action_key)
            invalid = observation.get("ok") is not True or bool(observation.get("error"))
            before = potential(
                covered_terms,
                discovered_documents,
                retrieved_documents,
                inspected_facets,
                checked_facets,
            )

            if not invalid and not repeated and name in {"search", "revise_query"}:
                query = str(arguments.get("query", ""))
                covered_terms.update(
                    term for term in required_terms if _matches_token(term, query)
                )
                returned = {
                    str(document_id)
                    for document_id in (observation.get("data") or {}).get("document_ids", [])
                }
                discovered_documents.update(returned & allowed_document_ids)
            elif not invalid and not repeated and name == "retrieve":
                document_id = str(arguments.get("document_id", ""))
                if document_id in allowed_document_ids:
                    discovered_documents.add(document_id)
                    retrieved_documents.add(document_id)
            elif not invalid and not repeated and name == "inspect_span":
                document_id = str(arguments.get("document_id", ""))
                document = document_map.get(document_id)
                span_id = str(arguments.get("span_id", ""))
                span = next(
                    (item for item in document.spans if item.span_id == span_id),
                    None,
                ) if document is not None else None
                facets = set(span.facets) & required_facets if span is not None else set()
                if document_id in allowed_document_ids:
                    discovered_documents.add(document_id)
                    retrieved_documents.add(document_id)
                inspected_facets.update(facets)
                handle_id = str((observation.get("data") or {}).get("handle_id", ""))
                if handle_id:
                    handle_facets[handle_id] = facets
            elif not invalid and not repeated and name == "check_citation":
                facets = handle_facets.get(str(arguments.get("handle_id", "")), set())
                checked_facets.update(facets)

            after = potential(
                covered_terms,
                discovered_documents,
                retrieved_documents,
                inspected_facets,
                checked_facets,
            )
            if invalid:
                credit = -1.0
                gain = "invalid_transition"
            elif repeated:
                credit = -0.5
                gain = "repeated_transition"
            elif name in TERMINAL_ACTIONS:
                credit = 1.0 if strict_success else -0.25
                gain = "strict_terminal_success" if strict_success else "terminal_failure"
            else:
                credit = round(after - before, 6)
                gain = "graph_potential_gain" if credit > 0.0 else "bridge_or_no_progress"
            milestone_counts[gain] += 1
            labels[(rollout_id, index)] = {
                "action": name,
                "credit": credit,
                "gain": gain,
                "potential_before": round(before, 6),
                "potential_after": round(after, 6),
                "invalid": invalid,
                "repeated": repeated,
            }
        rollout_final_potentials.append(
            potential(
                covered_terms,
                discovered_documents,
                retrieved_documents,
                inspected_facets,
                checked_facets,
            )
        )

    transformed = deepcopy(raw)
    positive = 0
    negative = 0
    zero = 0
    action_nonzero: dict[str, int] = defaultdict(int)
    for sample in transformed["trainer_batch"]["samples"]:
        key = (str(sample["rollout_id"]), int(sample["step_index"]))
        if key not in labels:
            raise ValueError(f"trainer sample has no graph-progress label: {key}")
        label = labels[key]
        sample["advantage"] = label["credit"]
        positive += label["credit"] > 0.0
        negative += label["credit"] < 0.0
        zero += label["credit"] == 0.0
        if label["credit"] != 0.0:
            action_nonzero[label["action"]] += 1

    method = "multihop_graph_progress_v6"
    transformed["trainer_batch"]["experiment_id"] = experiment_id
    transformed["trainer_batch"]["advantage_method"] = method
    trainer_batch_from_json(transformed["trainer_batch"])
    report = {
        "version": 6,
        "epistemic_status": "cpu_multihop_graph_progress_credit_not_model_training",
        "task_type": task_type,
        "samples": len(transformed["trainer_batch"]["samples"]),
        "eligible_anchors": 0,
        "eligible_samples": len(transformed["trainer_batch"]["samples"]),
        "nonzero_credit_samples": positive + negative,
        "positive_credit_samples": positive,
        "negative_credit_samples": negative,
        "zero_credit_samples": zero,
        "terminal_nonzero_credit_samples": sum(
            count for action, count in action_nonzero.items() if action in TERMINAL_ACTIONS
        ),
        "graph_potential_gain_samples": milestone_counts["graph_potential_gain"],
        "bridge_or_no_progress_samples": milestone_counts["bridge_or_no_progress"],
        "invalid_transition_samples": milestone_counts["invalid_transition"],
        "repeated_transition_samples": milestone_counts["repeated_transition"],
        "maximum_final_graph_potential": max(rollout_final_potentials, default=0.0),
        "mean_final_graph_potential": (
            sum(rollout_final_potentials) / len(rollout_final_potentials)
            if rollout_final_potentials else 0.0
        ),
        "nonzero_credit_by_action": dict(sorted(action_nonzero.items())),
        "gain_counts": dict(sorted(milestone_counts.items())),
        "utility_contract": "frozen_multihop_graph_potential_difference_with_terminal_credit",
    }
    transformed["credit_assignment"] = report
    return transformed, report
