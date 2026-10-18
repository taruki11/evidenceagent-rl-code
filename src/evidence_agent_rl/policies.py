"""Small non-learning policies used to validate the common harness."""

from __future__ import annotations

import re

from .harness import EvidenceHarness
from .models import Action, GuidanceKind, MemoryMode, TaskSpec


class FixedRAGPolicy:
    """Extractive lexical RAG without memory, RL, or hidden gold access."""

    name = "fixed_rag"

    def __init__(self, *, top_k: int = 1) -> None:
        self.top_k = top_k

    def run(self, harness: EvidenceHarness, task: TaskSpec, *, episode_id: str) -> None:
        harness.reset(task, episode_id=episode_id)
        search = harness.step(Action("search", {"query": task.question}))
        if harness.state.done:
            return
        candidates: list[tuple[str, str]] = []
        for document_id in search.data.get("document_ids", [])[: self.top_k]:
            if harness.state.done:
                break
            retrieved = harness.step(Action("retrieve", {"document_id": document_id}))
            if harness.state.done:
                return
            if not retrieved.ok:
                continue
            for span_id in retrieved.data["span_ids"]:
                inspected = harness.step(
                    Action(
                        "inspect_span",
                        {"document_id": document_id, "span_id": span_id},
                    )
                )
                if harness.state.done:
                    return
                if not inspected.ok:
                    continue
                handle_id = inspected.data["handle_id"]
                checked = harness.step(Action("check_citation", {"handle_id": handle_id}))
                if harness.state.done:
                    return
                if checked.ok:
                    candidates.append((inspected.data["text"], handle_id))
        if not candidates:
            harness.step(Action("abstain", {"reason": "No checked evidence in fixed top-k."}))
            return
        harness.step(
            Action(
                "answer",
                {
                    "claims": [
                        {"text": text, "evidence_handles": [handle_id]}
                        for text, handle_id in candidates
                    ]
                },
            )
        )


class OracleEvidencePolicy:
    """Hidden-gold upper bound for environment validation; never a model baseline."""

    name = "oracle_evidence"

    def run(self, harness: EvidenceHarness, task: TaskSpec, *, episode_id: str) -> None:
        _run_oracle(harness, task, episode_id=episode_id, use_memory=False)


class OracleEvidenceMemoryPolicy:
    """Gold teacher for SFT construction; never an inference baseline."""

    name = "oracle_evidence_memory"

    def run(self, harness: EvidenceHarness, task: TaskSpec, *, episode_id: str) -> None:
        _run_oracle(harness, task, episode_id=episode_id, use_memory=True)


class PublicExhaustionTeacher:
    """Insufficiency teacher whose decisions use public observations only."""

    name = "public_exhaustion_teacher"

    def run(self, harness: EvidenceHarness, task: TaskSpec, *, episode_id: str) -> None:
        harness.reset(task, episode_id=episode_id)
        _run_public_insufficiency_probe(harness)


class PublicMultiHopTeacher:
    """Two-hop teacher whose action choices use only public task text and observations."""

    name = "public_multihop_teacher"

    def run(self, harness: EvidenceHarness, task: TaskSpec, *, episode_id: str) -> None:
        if task.task_type != "multi_hop":
            raise ValueError("public multi-hop teacher requires a multi_hop task")
        harness.reset(task, episode_id=episode_id)

        statement_template = _question_section(
            task.question, "statement A:", "Then combine"
        )
        statement_query = re.sub(r"\[[A-Z_]+\]", " ", statement_template)
        first_search = harness.step(Action("search", {"query": statement_query}))
        first = _first_span_candidate(first_search.data.get("span_candidates", ()))
        if first is None:
            raise ValueError(f"public multi-hop search found no statement-A span: {task.task_id}")

        document_id, first_span_id = first
        if harness.policy_context_version in {"v4", "v5", "v6"}:
            first_evidence = harness.step(
                Action(
                    "select_evidence",
                    {"document_id": document_id, "span_id": first_span_id},
                )
            )
        else:
            retrieved = harness.step(Action("retrieve", {"document_id": document_id}))
            if not retrieved.ok:
                raise ValueError(f"public multi-hop document retrieval failed: {task.task_id}")
            first_evidence = harness.step(
                Action(
                    "inspect_span",
                    {"document_id": document_id, "span_id": first_span_id},
                )
            )
        if not first_evidence.ok:
            raise ValueError(f"public multi-hop statement-A inspection failed: {task.task_id}")
        first_handle = str(first_evidence.data["handle_id"])
        if harness.policy_context_version not in {"v4", "v5", "v6"}:
            harness.step(Action("check_citation", {"handle_id": first_handle}))

        observed_statement = str(first_evidence.data["text"])
        recovered_values = _recover_masked_values(statement_template, observed_statement)
        selector_query = (
            _question_section(task.question, "selector terms", "to locate").strip(" ,")
            if "selector terms" in task.question.casefold()
            else ""
        )
        if selector_query:
            second_query = " ".join([*recovered_values, selector_query]).strip()
        else:
            bridge_values = _recover_masked_values(
                statement_template,
                observed_statement,
                mask_name="BRIDGE",
            )
            if len(bridge_values) != 1:
                raise ValueError(
                    "selector-free multi-hop question requires exactly one [BRIDGE] mask"
                )
            second_query = bridge_values[0]
        second_search = harness.step(
            Action("search", {"query": second_query})
        )
        second = _first_span_candidate(
            second_search.data.get("span_candidates", ()),
            document_id=document_id,
            excluded_span_ids={first_span_id},
        )
        if second is None:
            raise ValueError(f"public multi-hop search found no second span: {task.task_id}")

        _, second_span_id = second
        second_evidence = harness.step(
            Action(
                (
                    "select_evidence"
                    if harness.policy_context_version in {"v4", "v5", "v6"}
                    else "inspect_span"
                ),
                {"document_id": document_id, "span_id": second_span_id},
            )
        )
        if not second_evidence.ok:
            raise ValueError(f"public multi-hop second-span inspection failed: {task.task_id}")
        second_handle = str(second_evidence.data["handle_id"])
        if harness.policy_context_version not in {"v4", "v5", "v6"}:
            harness.step(Action("check_citation", {"handle_id": second_handle}))
        harness.step(
            Action(
                "answer",
                {
                    "claims": [
                        {
                            "text": observed_statement,
                            "evidence_handles": [first_handle],
                        },
                        {
                            "text": str(second_evidence.data["text"]),
                            "evidence_handles": [second_handle],
                        },
                    ]
                },
            )
        )


def _question_section(question: str, start: str, end: str) -> str:
    pattern = rf"{re.escape(start)}\s*(.*?)\s*{re.escape(end)}"
    match = re.search(pattern, question, flags=re.IGNORECASE | re.DOTALL)
    if match is None or not match.group(1).strip():
        raise ValueError(f"multi-hop question lacks public section {start!r} ... {end!r}")
    return match.group(1).strip()


def _recover_masked_values(
    template: str,
    observed: str,
    *,
    mask_name: str | None = None,
) -> tuple[str, ...]:
    normalized_template = " ".join(template.split())
    normalized_observed = " ".join(observed.split())
    masks = tuple(re.findall(r"\[(BRIDGE|BLANK)\]", normalized_template))
    parts = re.split(r"\[(?:BRIDGE|BLANK)\]", normalized_template)
    if len(parts) < 2:
        raise ValueError("multi-hop statement A has no public masks")
    pattern = "^" + "(.+?)".join(re.escape(part) for part in parts) + "$"
    match = re.match(pattern, normalized_observed, flags=re.IGNORECASE)
    if match is None:
        raise ValueError("observed statement A does not match its public masked template")
    recovered = tuple(value.strip() for value in match.groups())
    values = tuple(
        value
        for mask, value in zip(masks, recovered, strict=True)
        if value and (mask_name is None or mask == mask_name)
    )
    if not values:
        raise ValueError("multi-hop statement A produced no recovered values")
    return values


def _first_span_candidate(
    candidates: object,
    *,
    document_id: str | None = None,
    excluded_span_ids: set[str] | None = None,
) -> tuple[str, str] | None:
    excluded = excluded_span_ids or set()
    if not isinstance(candidates, (list, tuple)):
        return None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_document = candidate.get("document_id")
        candidate_span = candidate.get("span_id")
        if not isinstance(candidate_document, str) or not isinstance(candidate_span, str):
            continue
        if document_id is not None and candidate_document != document_id:
            continue
        if candidate_span in excluded:
            continue
        return candidate_document, candidate_span
    return None


def _run_oracle(
    harness: EvidenceHarness,
    task: TaskSpec,
    *,
    episode_id: str,
    use_memory: bool,
) -> None:
    harness.reset(task, episode_id=episode_id)
    if (
        use_memory
        and harness.memory_mode is not MemoryMode.NONE
        and task.available_memory_ids
    ):
        memory_id = task.available_memory_ids[0]
        inspected = harness.step(Action("inspect_memory", {"memory_id": memory_id}))
        if harness.state.done:
            return
        if inspected.ok:
            if harness.memory_mode is MemoryMode.EVIDENCE_BOUND_RECONSTRUCTION:
                if task.mismatch_role == "mismatched":
                    kind = GuidanceKind.EMPTY.value
                    instruction = ""
                else:
                    historical = inspected.data["historical_strategy"]
                    kind = historical["kind"]
                    instruction = historical["instruction"]
                harness.step(
                    Action(
                        "reconstruct_memory",
                        {"memory_id": memory_id, "kind": kind, "instruction": instruction},
                    )
                )
                if harness.state.done:
                    return
    selected = [
        document
        for document in harness.documents.values()
        if document.doi in task.allowed_dois
    ]
    candidates: list[tuple[str, str]] = []
    if selected:
        if harness.policy_context_version in {"v4", "v5", "v6"}:
            # The macro teacher may only select public span candidates.  Use the
            # task wording as the query so the gold span is surfaced by lexical
            # search; never inject the hidden answer key into the query.
            query = " ".join([task.question, *task.required_query_terms])
        else:
            query = " ".join(
                [*(doc.title for doc in selected), *task.required_query_terms]
            )
        harness.step(Action("search", {"query": query}))
        if harness.state.done:
            return
    for document in selected:
        if harness.policy_context_version not in {"v4", "v5", "v6"}:
            retrieved = harness.step(Action("retrieve", {"document_id": document.document_id}))
            if harness.state.done:
                return
            if not retrieved.ok:
                continue
        for span in document.spans:
            if not (set(span.facets) & set(task.required_facets)):
                continue
            inspected = harness.step(
                Action(
                    (
                        "select_evidence"
                        if harness.policy_context_version in {"v4", "v5", "v6"}
                        else "inspect_span"
                    ),
                    {"document_id": document.document_id, "span_id": span.span_id},
                )
            )
            if harness.state.done:
                return
            if not inspected.ok:
                continue
            handle_id = inspected.data["handle_id"]
            if harness.policy_context_version in {"v4", "v5", "v6"}:
                candidates.append((inspected.data["text"], handle_id))
            else:
                checked = harness.step(Action("check_citation", {"handle_id": handle_id}))
                if harness.state.done:
                    return
                if checked.ok:
                    candidates.append((inspected.data["text"], handle_id))
    if task.insufficient:
        harness.step(Action("abstain", {"reason": "Oracle gold marks evidence insufficient."}))
    elif candidates:
        harness.step(
            Action(
                "answer",
                {
                    "claims": [
                        {"text": text, "evidence_handles": [handle_id]}
                        for text, handle_id in candidates
                    ]
                },
            )
        )
    else:
        harness.step(Action("abstain", {"reason": "Oracle found no gold evidence."}))


def _run_public_insufficiency_probe(
    harness: EvidenceHarness,
) -> None:
    """Build an abstention trajectory justified by public observations.

    The dataset builder may select this teacher using a reviewed task label,
    but the teacher itself reads only the same public state and observations as
    the student. It follows search/retrieve/inspect until the budget or the
    public candidate spans are exhausted before abstaining.
    """

    public_state = harness.render_policy_state()
    searched = harness.step(Action("search", {"query": public_state["question"]}))
    if harness.state.done:
        return
    document_ids = searched.data.get("document_ids", []) if searched.ok else []
    if not document_ids:
        harness.step(
            Action(
                "abstain",
                {"reason": "Search returned no current-corpus candidates."},
            )
        )
        return

    if harness.policy_context_version in {"v4", "v5", "v6"}:
        for candidate in searched.data.get("span_candidates", []):
            if harness.state.remaining_budget < 1:
                break
            selected = harness.step(
                Action(
                    "select_evidence",
                    {
                        "document_id": candidate["document_id"],
                        "span_id": candidate["span_id"],
                    },
                )
            )
            if harness.state.done:
                return
            if not selected.ok:
                continue
        harness.step(
            Action(
                "abstain",
                {
                    "reason": (
                        "The searched current-episode evidence did not support "
                        "the requested attribute."
                    )
                },
            )
        )
        return

    for document_id in document_ids:
        if harness.state.remaining_budget < 1:
            break
        retrieved = harness.step(Action("retrieve", {"document_id": document_id}))
        if harness.state.done:
            return
        if not retrieved.ok:
            continue
        for span_id in retrieved.data.get("span_ids", []):
            if harness.state.remaining_budget < 1:
                break
            inspected = harness.step(
                Action(
                    "inspect_span",
                    {"document_id": document_id, "span_id": span_id},
                )
            )
            if harness.state.done:
                return
            if not inspected.ok:
                continue

    harness.step(
        Action(
            "abstain",
            {
                "reason": (
                    "The available search and inspected current-episode spans "
                    "did not support the requested attribute."
                )
            },
        )
    )
