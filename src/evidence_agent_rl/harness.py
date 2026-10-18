"""State-externalizing harness with an enforceable evidence firewall."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import replace
from typing import Any

from .hotpotqa_answer import (
    ANSWER_PROTOCOL_SHORT,
    ANSWER_PROTOCOLS,
    short_answer_argument_schema,
)
from .models import (
    Action,
    Claim,
    Document,
    EpisodeState,
    EvidenceHandle,
    Guidance,
    GuidanceKind,
    MemoryMode,
    Observation,
    Span,
    StrategyMemory,
    TaskSpec,
    TraceStep,
)
from .skills import (
    SPAN_SCORING_MODES,
    Skill,
    _tokens,
    build_skill_registry,
    validate_arguments_against,
)


class EvidenceHarness:
    def __init__(
        self,
        documents: Iterable[Document],
        memories: Iterable[StrategyMemory] = (),
        *,
        max_budget: int = 8,
        memory_mode: MemoryMode | str = MemoryMode.EVIDENCE_BOUND_RECONSTRUCTION,
        policy_context_version: str = "v2",
        sequential_multihop_gate: bool = False,
        public_bridge_query_controller: bool = False,
        document_scoped_followup_search: bool = False,
        span_scoring: str = "overlap",
        selectable_band: int = 0,
        selectable_protect_bridge: bool = False,
        answer_protocol: str = "atomic_claims",
        require_distinct_support_documents: bool = False,
    ) -> None:
        if max_budget <= 0:
            raise ValueError("max_budget must be positive")
        if policy_context_version not in {"v1", "v2", "v3", "v4", "v5", "v6"}:
            raise ValueError("policy_context_version must be v1, v2, v3, v4, v5, or v6")
        # The answer protocol is versioned separately from ``policy_context_version`` so that the
        # historical behaviour stays byte-for-byte the default: HotpotQA needs to separate the
        # short answer from the supporting set, which the atomic-claims schema cannot express.
        if answer_protocol not in ANSWER_PROTOCOLS:
            raise ValueError(f"answer_protocol must be one of {ANSWER_PROTOCOLS}")
        if not isinstance(require_distinct_support_documents, bool):
            raise TypeError("require_distinct_support_documents must be a boolean")
        if not isinstance(sequential_multihop_gate, bool):
            raise TypeError("sequential_multihop_gate must be a boolean")
        if sequential_multihop_gate and policy_context_version != "v6":
            raise ValueError("sequential_multihop_gate requires policy_context_version v6")
        if not isinstance(public_bridge_query_controller, bool):
            raise TypeError("public_bridge_query_controller must be a boolean")
        if public_bridge_query_controller and not sequential_multihop_gate:
            raise ValueError(
                "public_bridge_query_controller requires sequential_multihop_gate"
            )
        if not isinstance(document_scoped_followup_search, bool):
            raise TypeError("document_scoped_followup_search must be a boolean")
        if document_scoped_followup_search and not sequential_multihop_gate:
            raise ValueError(
                "document_scoped_followup_search requires sequential_multihop_gate"
            )
        if span_scoring not in SPAN_SCORING_MODES:
            raise ValueError(
                f"span_scoring must be one of {SPAN_SCORING_MODES}, got {span_scoring!r}"
            )
        if not isinstance(selectable_band, int) or isinstance(selectable_band, bool):
            raise TypeError("selectable_band must be an integer")
        if selectable_band < 0:
            raise ValueError("selectable_band must be >= 0 (0 disables the band)")
        if not isinstance(selectable_protect_bridge, bool):
            raise TypeError("selectable_protect_bridge must be a boolean")
        if selectable_protect_bridge and selectable_band <= 0:
            raise ValueError(
                "selectable_protect_bridge requires selectable_band > 0; it widens a band, so "
                "there is nothing for it to protect inside band 0 (the frozen full frontier)"
            )
        self.documents = {document.document_id: document for document in documents}
        self.memories = {memory.memory_id: memory for memory in memories}
        self.max_budget = max_budget
        self.memory_mode = MemoryMode(memory_mode)
        self.policy_context_version = policy_context_version
        self.answer_protocol = answer_protocol
        # Declared public structural requirement: the submitted support spans must come from
        # two different documents.  Off by default so no earlier behaviour moves.
        self.require_distinct_support_documents = require_distinct_support_documents
        self.sequential_multihop_gate = sequential_multihop_gate
        self.public_bridge_query_controller = public_bridge_query_controller
        self.document_scoped_followup_search = document_scoped_followup_search
        self.span_scoring = span_scoring
        # ``selectable_band`` > 0 restricts the hop-2 CHOICE SET to the top-scoring entries of
        # the document-scoped window, WITHOUT changing the window itself (so the search
        # instrument and the frozen frontier membership are untouched) and WITHOUT changing
        # the rendered order (the band members are still rendered alphabetically, exactly as
        # before).  0 is the frozen behaviour byte for byte.
        #
        # Rationale (`tmp/p31`, `tmp/p32`): the hop-2 pick's dense score-rank is bimodal --
        # 21/48 decisions at rank 0 and 10/48 at rank >= 12 -- and every miss lives in the
        # deep mode, where the policy reaches into a 22-candidate list and takes a span it
        # prefers on content. A narrow band removes that mode: the content attractor
        # `auto-2607.28573-4ec3dcce4f5ba243-span` is in the top-3 in 0/48 decisions while
        # being offered in 24/48 full frontiers, and gold is in the top-3 in 22/48.
        self.selectable_band = selectable_band
        # ``selectable_protect_bridge`` widens a non-zero band to also admit the spans carrying
        # the task's declared bridge term (``required_query_terms[0]``).  It exists because
        # `tmp/p38` showed that every exclusion a top-3 band caused on the frozen canary was a
        # strict score-MAGNITUDE exclusion -- no tie-break, rank-ordinal or normalisation change
        # could recover those tasks, so only a membership rule can.  The bridge term is a signal
        # the policy already received (it is inside the frozen forced query where it applies), so
        # this protects evidence without consulting gold.  Measured on the development tasks it
        # retained the hop-2 target 8/8 while never re-admitting the observed content attractor.
        self.selectable_protect_bridge = selectable_protect_bridge
        self.skills: dict[str, Skill] = build_skill_registry()
        self.state: EpisodeState

    def reset(self, task: TaskSpec, *, episode_id: str) -> EpisodeState:
        if not episode_id.strip():
            raise ValueError("episode_id is required for deterministic replay")
        self.state = EpisodeState(episode_id, task, self.max_budget)
        self.public_span_candidates: set[tuple[str, str]] = set()
        self.public_span_metadata: dict[tuple[str, str], dict[str, Any]] = {}
        # The corpus is immutable, but the task's ``corpus_scope`` is not, so the
        # IDF table must not leak across episodes.
        self._idf_cache = None
        return self.state

    def step(self, action: Action) -> Observation:
        if not hasattr(self, "state"):
            raise RuntimeError("reset the harness before stepping")
        if self.state.done:
            raise RuntimeError("episode has already terminated")
        if action.name == "__provider_error__":
            error = action.arguments.get("error")
            if set(action.arguments) != {"error"} or not isinstance(error, str) or not error:
                observation = Observation(
                    False, "invalid_action", error="invalid provider error action"
                )
                return self._record(action, observation)
            self.state.done = True
            return self._record(
                action,
                Observation(False, "provider_error", error=error),
            )
        if action.name == "__decision_limit__":
            if action.arguments:
                return self._record(
                    action,
                    Observation(False, "invalid_action", error="decision limit takes no arguments"),
                )
            self.state.done = True
            return self._record(
                action,
                Observation(False, "decision_limit_exhausted", error="policy decision limit exhausted"),
            )
        skill = self.skills.get(action.name)
        if skill is None:
            observation = Observation(False, "invalid_action", error=f"unknown skill: {action.name}")
            return self._record(action, observation)
        if (
            self.policy_context_version not in {"v3", "v4", "v5", "v6"}
            and action.name not in self._available_skill_names()
        ):
            observation = Observation(False, "invalid_action", error=f"unknown skill: {action.name}")
            return self._record(action, observation)
        argument_error = self._validate_action_arguments(skill, action)
        if argument_error:
            if (
                self.policy_context_version in {"v5", "v6"}
                or self.answer_protocol == ANSWER_PROTOCOL_SHORT
            ) and action.name == "answer":
                self.state.done = True
            observation = Observation(False, "invalid_action", error=argument_error)
            return self._record(action, observation)
        if self.policy_context_version in {"v3", "v4", "v5", "v6"}:
            progress_error = self._v3_progress_error(action)
            if progress_error is not None:
                return self._record(
                    action,
                    Observation(False, "no_progress_action", error=progress_error),
                )
        if skill.cost > self.state.remaining_budget:
            if self.policy_context_version in {"v3", "v4", "v5", "v6"}:
                return self._record(
                    action,
                    Observation(
                        False,
                        "action_unavailable",
                        error="action cost exceeds remaining budget; answer or abstain",
                    ),
                )
            self.state.done = True
            observation = Observation(False, "budget_exhausted", error="tool budget exhausted")
            return self._record(action, observation)
        if (
            self.policy_context_version in {"v3", "v4", "v5", "v6"}
            and action.name not in self._available_skill_names()
        ):
            if self.policy_context_version in {"v5", "v6"} and action.name == "answer":
                self.state.done = True
            observation = Observation(False, "invalid_action", error=f"unknown skill: {action.name}")
            return self._record(action, observation)
        self.state.remaining_budget -= skill.cost
        try:
            observation = skill.handler(self, action.arguments)
        except (TypeError, ValueError) as exc:
            observation = Observation(False, action.name, error=str(exc))
        return self._record(action, observation)

    def _v3_progress_error(self, action: Action) -> str | None:
        """Reject repeated paid transitions before they consume budget."""

        if action.name == "search":
            query = action.arguments.get("query")
            if (
                self.public_bridge_query_controller
                and self.fresh_multihop_search_required()
                and query != self.public_bridge_query()
            ):
                return "search query must equal the current public bridge query"
            if isinstance(query, str) and query.strip() in self.state.query_history:
                return "search query was already issued"
        elif action.name == "retrieve":
            document_id = action.arguments.get("document_id")
            if (
                isinstance(document_id, str)
                and document_id in self.state.retrieved_documents
            ):
                return "document was already retrieved"
        elif action.name == "inspect_span":
            document_id = action.arguments.get("document_id")
            span_id = action.arguments.get("span_id")
            if any(
                handle.document_id == document_id and handle.span_id == span_id
                for handle in self.state.evidence.values()
            ):
                return "span was already inspected"
        elif action.name == "check_citation":
            handle_id = action.arguments.get("handle_id")
            if (
                isinstance(handle_id, str)
                and handle_id in self.state.verification_records
            ):
                return "evidence handle was already citation-checked"
        elif action.name == "select_evidence":
            document_id = action.arguments.get("document_id")
            span_id = action.arguments.get("span_id")
            if any(
                handle.document_id == document_id and handle.span_id == span_id
                for handle in self.state.evidence.values()
            ):
                return "span was already selected"
        return None

    def _record(self, action: Action, observation: Observation) -> Observation:
        self.state.trace.append(
            TraceStep(len(self.state.trace), action, observation, self.state.remaining_budget)
        )
        return observation

    def make_evidence_handle(self, document: Document, span: Span) -> EvidenceHandle:
        handle_id = ":".join(
            [self.state.episode_id, self.state.task.task_id, document.document_id, span.span_id]
        )
        return EvidenceHandle(
            handle_id,
            self.state.episode_id,
            self.state.task.task_id,
            document.document_id,
            span.span_id,
            document.doi,
            span.text,
            span.facets,
        )

    def _validate_action_arguments(self, skill: Any, action: Action) -> str | None:
        """Validate an action's arguments against the schema that is actually in force.

        The `answer` action's schema depends on the declared answer protocol, so the runtime check
        must use the same definition the public catalog rendered -- otherwise the model would be
        shown one contract and held to another.
        """
        if action.name == "answer" and self.answer_protocol == ANSWER_PROTOCOL_SHORT:
            return validate_arguments_against(
                short_answer_argument_schema(self.minimum_answer_evidence_handles()),
                action.arguments,
            )
        return skill.validate_arguments(action.arguments)

    @staticmethod
    def parse_claims(raw_claims: Any) -> tuple[Claim, ...]:
        if not isinstance(raw_claims, list) or not raw_claims:
            raise ValueError("claims must be a non-empty list")
        claims: list[Claim] = []
        for raw in raw_claims:
            if not isinstance(raw, dict):
                raise ValueError("each claim must be an object")
            handles = raw.get("evidence_handles")
            if not isinstance(handles, list) or not handles:
                raise ValueError("each claim requires evidence_handles")
            claims.append(
                Claim(
                    text=str(raw.get("text", "")).strip(),
                    evidence_handles=tuple(str(item) for item in handles),
                )
            )
        if any(not claim.text for claim in claims):
            raise ValueError("claim text is required")
        return tuple(claims)

    def validate_answer_evidence(self, claims: tuple[Claim, ...]) -> str | None:
        for claim in claims:
            for handle_id in claim.evidence_handles:
                handle = self.state.evidence.get(handle_id)
                if handle is None:
                    return f"foreign or historical evidence handle: {handle_id}"
                if handle.episode_id != self.state.episode_id or handle.task_id != self.state.task.task_id:
                    return f"evidence handle is not bound to this episode: {handle_id}"
                if not handle.checked:
                    return f"evidence handle was not citation-checked: {handle_id}"
        return None

    def minimum_answer_evidence_handles(self) -> int:
        """Return the public task contract's minimum checked support count."""

        if not hasattr(self, "state"):
            return 1
        return 2 if self.state.task.task_type == "multi_hop" else 1

    def fresh_multihop_search_required(self) -> bool:
        """Require a public search after the first multi-hop evidence selection."""

        if (
            not self.sequential_multihop_gate
            or not hasattr(self, "state")
            or self.state.task.task_type != "multi_hop"
            or len(self.state.evidence) != 1
        ):
            return False
        first_selection = next(
            (
                step.index
                for step in self.state.trace
                if step.action.name == "select_evidence" and step.observation.ok
            ),
            None,
        )
        if first_selection is None:
            return False
        return not any(
            step.index > first_selection
            and step.action.name == "search"
            and step.observation.ok
            for step in self.state.trace
        )

    def public_bridge_query(self) -> str | None:
        """Build a deterministic follow-up query from current public evidence."""

        if not self.fresh_multihop_search_required() or len(self.state.evidence) != 1:
            return None
        handle = next(iter(self.state.evidence.values()))
        return " ".join(f"{self.state.task.question} {handle.text}".split())

    def followup_search_document_id(self) -> str | None:
        """Return the single document the next follow-up search must be scoped to.

        ``None`` means "no document scope": search the whole corpus, which is the
        historical behaviour.  A non-``None`` value means the harness has been
        constructed with ``document_scoped_followup_search=True`` and the sequential
        multi-hop condition is currently armed, so the next ``search`` may only scan
        the document of the single current-episode evidence handle.

        The document identifier is already public: it is rendered in the evidence
        ledger and in the selectable-evidence frontier.  No hidden ``TaskSpec`` field
        (``required_facets``, ``deterministic_answer_key``, ``required_query_terms``,
        ``allowed_dois``, ``mismatch_role``) is read here.
        """

        if not self.document_scoped_followup_search or not hasattr(self, "state"):
            return None
        if self.policy_context_version != "v6" or not self.sequential_multihop_gate:
            return None
        if self.state.task.task_type != "multi_hop":
            return None
        if len(self.state.evidence) != 1:
            return None
        if not self.fresh_multihop_search_required():
            return None
        handle = next(iter(self.state.evidence.values()))
        return handle.document_id

    def replay(self, task: TaskSpec, actions: Iterable[Action], *, episode_id: str) -> EpisodeState:
        self.reset(task, episode_id=episode_id)
        for action in actions:
            if self.state.done:
                break
            self.step(action)
        return self.state

    def cloned_action_trace(self) -> tuple[Action, ...]:
        return tuple(replace(step.action, arguments=dict(step.action.arguments)) for step in self.state.trace)

    def render_policy_state(self) -> dict[str, Any]:
        """Return compact harness-owned bookkeeping for the policy context."""
        if not hasattr(self, "state"):
            raise RuntimeError("reset the harness before rendering state")
        ranked_candidates = sorted(
            self.state.candidate_pool.items(), key=lambda item: (-item[1], item[0])
        )
        rendered = {
            "task_id": self.state.task.task_id,
            "question": self.state.task.question,
            "query_history": list(self.state.query_history),
            "candidate_pool": [
                {"document_id": document_id, "score": score}
                for document_id, score in ranked_candidates
            ],
            "retrieved_documents": sorted(self.state.retrieved_documents),
            "verification_records": dict(sorted(self.state.verification_records.items())),
            "evidence_handles": sorted(self.state.evidence),
            "inspected_memory_ids": sorted(self.state.inspected_memory_ids),
            "available_memory_ids": (
                []
                if self.memory_mode is MemoryMode.NONE
                else (
                    list(self.state.task.available_memory_ids)
                    if self.state.task.available_memory_ids is not None
                    else sorted(self.memories)
                )
            ),
            "memory_mode": self.memory_mode.value,
            "remaining_budget": self.state.remaining_budget,
            "done": self.state.done,
        }
        if self.sequential_multihop_gate:
            rendered["sequential_multihop_gate"] = {
                "enabled": True,
                "fresh_search_required_before_next_selection": (
                    self.fresh_multihop_search_required()
                ),
            }
            if self.public_bridge_query_controller:
                rendered["sequential_multihop_gate"]["required_public_bridge_query"] = (
                    self.public_bridge_query()
                )
            # The active document scope is deliberately NOT rendered here.  Even though the
            # document identifier is derivable from the public ledger, emitting an explicit
            # hint would change the prompt and turn a retrieval-pool intervention into a
            # prompt intervention, breaking the single-variable design.  The scope actually
            # applied is recorded machine-side instead: every search observation carries a
            # ``scoped_search`` block, which v3-v6 strip from the model-visible context.
        if self.policy_context_version == "v2":
            rendered["memory_decisions"] = [
                {
                    "memory_id": guidance.memory_id,
                    "kind": guidance.kind.value,
                    "instruction": guidance.instruction,
                }
                for guidance in self.state.guidance_history
            ]
        if self.policy_context_version in {"v3", "v4", "v5", "v6"}:
            rendered["memory_decisions"] = [
                {
                    "memory_id": guidance.memory_id,
                    "kind": guidance.kind.value,
                    "instruction": guidance.instruction,
                }
                for guidance in self.state.guidance_history
            ]
            rendered["retrieved_document_ledger"] = [
                {
                    "document_id": document.document_id,
                    "title": document.title,
                    "doi": document.doi,
                    "document_type": document.document_type,
                    "span_ids": [span.span_id for span in document.spans],
                }
                for document in sorted(
                    (
                        self.documents[document_id]
                        for document_id in self.state.retrieved_documents
                    ),
                    key=lambda item: item.document_id,
                )
            ]
            rendered["evidence_ledger"] = [
                {
                    "handle_id": handle.handle_id,
                    "document_id": handle.document_id,
                    "span_id": handle.span_id,
                    "text": handle.text,
                    "citation_checked": self.state.verification_records.get(
                        handle.handle_id
                    ),
                }
                for handle in sorted(
                    self.state.evidence.values(), key=lambda item: item.handle_id
                )
            ]
            paid_actions_available = self.state.remaining_budget >= 1
            rendered["action_frontier"] = {
                "retrievable_document_ids": (
                    sorted(set(self.state.candidate_pool) - self.state.retrieved_documents)
                    if paid_actions_available
                    else []
                ),
                "inspectable_spans": [
                    {"document_id": document_id, "span_id": span.span_id}
                    for document_id in sorted(self.state.retrieved_documents)
                    for span in self.documents[document_id].spans
                    if paid_actions_available
                    and not any(
                        handle.document_id == document_id
                        and handle.span_id == span.span_id
                        for handle in self.state.evidence.values()
                    )
                ],
                "unchecked_evidence_handles": (
                    sorted(set(self.state.evidence) - set(self.state.verification_records))
                    if paid_actions_available
                    else []
                ),
            }
            if self.policy_context_version in {"v4", "v5", "v6"}:
                selected = {
                    (handle.document_id, handle.span_id)
                    for handle in self.state.evidence.values()
                }
                rendered["action_frontier"] = {
                    "selectable_evidence": [
                        {
                            "document_id": document_id,
                            "span_id": span_id,
                            **(
                                self.public_span_metadata[(document_id, span_id)]
                                if self.policy_context_version == "v6"
                                else {}
                            ),
                        }
                        for document_id, span_id in self.selectable_keys()
                        if paid_actions_available
                        and not self.fresh_multihop_search_required()
                    ]
                }
                if self.policy_context_version in {"v5", "v6"}:
                    rendered["termination_requirements"] = {
                        **(
                            {"require_distinct_support_documents": True}
                            if self.require_distinct_support_documents
                            else {}
                        ),
                                    "minimum_distinct_checked_evidence_handles": (
                            self.minimum_answer_evidence_handles()
                        ),
                        "search_required_before_abstention": True,
                        "answer_submission_is_terminal": True,
                        **(
                            {
                                "fresh_search_required_before_next_selection": (
                                    self.fresh_multihop_search_required()
                                )
                            }
                            if self.sequential_multihop_gate
                            else {}
                        ),
                    }
        return rendered

    def selectable_keys(self) -> list[tuple[str, str]]:
        """The drawable evidence handles, in the frozen alphabetical render order.

        With ``selectable_band == 0`` this is exactly
        ``sorted(public_span_candidates - selected)`` -- the frozen behaviour, byte for byte.

        With ``selectable_band == k > 0`` the set is narrowed to the ``k`` highest-scoring
        members of that same set, but the returned ORDER is still alphabetical, so the change
        is to *membership* only and never to the presentation order.  Ties at the cut-off are
        broken by the same ``(-score, document_id, span_id)`` key the scorer uses, so the band
        is deterministic.
        """

        selected = {
            (handle.document_id, handle.span_id)
            for handle in self.state.evidence.values()
        }
        eligible = sorted(self.public_span_candidates - selected)
        if self.selectable_band <= 0 or len(eligible) <= self.selectable_band:
            return eligible
        ranked = sorted(
            eligible,
            key=lambda key: (
                -float(self.public_span_metadata.get(key, {}).get("score", 0.0)),
                key[0],
                key[1],
            ),
        )
        chosen = list(ranked[: self.selectable_band])
        if self.selectable_protect_bridge:
            chosen.extend(self._bridge_carriers(eligible))
        return sorted({key: None for key in chosen}.keys())

    def _bridge_carriers(self, eligible: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Eligible spans whose text carries the task's declared bridge term.

        The term is taken from ``task.required_query_terms`` (the same task-declared link term the
        `idf_max_bridge` scorer uses) and matched with the harness's own tokenizer, so the
        protection cannot drift from how the query is tokenised.  An empty term set protects
        nothing rather than everything.
        """

        terms = getattr(self.state.task, "required_query_terms", ()) or ()
        folded = [str(t).casefold() for t in terms if str(t).strip()]
        if not folded:
            return []
        carriers = []
        for key in eligible:
            text = self._span_text(key)
            tokens = _tokens(text) if text else set()
            if any(term in tokens for term in folded):
                carriers.append(key)
        return carriers

    def _span_text(self, key: tuple[str, str]) -> str | None:
        document = self.documents.get(key[0])
        if document is None:
            return None
        for span in document.spans:
            if span.span_id == key[1]:
                return span.text
        return None

    def skill_catalog(self) -> tuple[dict[str, Any], ...]:
        available = self._available_skill_names()
        catalog = []
        for skill in self.skills.values():
            if skill.name not in available:
                continue
            spec = deepcopy(skill.public_spec())
            if (
                skill.name == "search"
                and self.public_bridge_query_controller
                and self.fresh_multihop_search_required()
            ):
                query = self.public_bridge_query()
                if query is None:
                    raise RuntimeError("public bridge query is unavailable")
                spec["parameters"] = {
                    "type": "object",
                    "properties": {"query": {"type": "string", "enum": [query]}},
                    "required": ["query"],
                    "additionalProperties": False,
                }
            if (
                skill.name == "answer"
                and self.answer_protocol == ANSWER_PROTOCOL_SHORT
            ):
                # HotpotQA: the answer string and the supporting set are separate fields, and the
                # model may submit any number of handles (including none).  Nothing here is derived
                # from gold and nothing is auto-cited from the evidence ledger.
                spec["parameters"] = short_answer_argument_schema(
                    self.minimum_answer_evidence_handles()
                )
            elif self.policy_context_version in {"v5", "v6"} and skill.name == "answer":
                claims = spec["parameters"]["properties"]["claims"]
                claims["minItems"] = self.minimum_answer_evidence_handles()
                handles = claims["items"]["properties"]["evidence_handles"]
                handles["minItems"] = 1
                handles["maxItems"] = 1
            if (
                self.policy_context_version == "v6"
                and skill.name == "select_evidence"
                and hasattr(self, "state")
            ):
                selectable = self.selectable_keys()
                spans_by_document: dict[str, list[str]] = {}
                for document_id, span_id in selectable:
                    spans_by_document.setdefault(document_id, []).append(span_id)
                spec["parameters"] = {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "document_id": {"type": "string", "const": document_id},
                                "span_id": {"type": "string", "enum": span_ids},
                            },
                            "required": ["document_id", "span_id"],
                            "additionalProperties": False,
                        }
                        for document_id, span_ids in spans_by_document.items()
                    ]
                }
            catalog.append(spec)
        return tuple(catalog)

    def _available_skill_names(self) -> frozenset[str]:
        if self.policy_context_version in {"v5", "v6"}:
            macro_actions = {"search", "select_evidence", "answer", "abstain"}
            if not hasattr(self, "state"):
                return frozenset(macro_actions)
            if self.fresh_multihop_search_required():
                if self.state.remaining_budget >= self.skills["search"].cost:
                    return frozenset({"search"})
                return frozenset({"abstain"})
            available: set[str] = set()
            if self.state.remaining_budget >= self.skills["search"].cost:
                available.add("search")
            selected = {
                (handle.document_id, handle.span_id)
                for handle in self.state.evidence.values()
            }
            selectable = self.public_span_candidates - selected
            if (
                selectable
                and self.state.remaining_budget >= self.skills["select_evidence"].cost
            ):
                available.add("select_evidence")
            checked_handles = {
                handle_id
                for handle_id, checked in self.state.verification_records.items()
                if checked
            }
            if len(checked_handles) >= self.minimum_answer_evidence_handles():
                if not self.require_distinct_support_documents:
                    available.add("answer")
                elif len(
                    {
                        self.state.evidence[h].document_id
                        for h in checked_handles
                        if h in self.state.evidence
                    }
                ) >= 2:
                    available.add("answer")
            if self.state.query_history and (
                self.state.remaining_budget == 0 or not selectable
            ):
                available.add("abstain")
            return frozenset(available)
        if self.policy_context_version == "v4":
            base = {"search", "select_evidence", "answer", "abstain"}
            if hasattr(self, "state"):
                base = {
                    name
                    for name in base
                    if self.skills[name].cost <= self.state.remaining_budget
                }
            return frozenset(base)
        base = set(self.skills) - {"inspect_memory", "reconstruct_memory", "select_evidence"}
        if self.memory_mode is MemoryMode.NONE:
            if self.policy_context_version == "v3" and hasattr(self, "state"):
                base = {
                    name
                    for name in base
                    if self.skills[name].cost <= self.state.remaining_budget
                }
            return frozenset(base)
        if self.policy_context_version == "v1":
            base.add("inspect_memory")
            if self.memory_mode is MemoryMode.EVIDENCE_BOUND_RECONSTRUCTION:
                base.add("reconstruct_memory")
            return frozenset(base)
        if not hasattr(self, "state"):
            base.add("inspect_memory")
            if self.memory_mode is MemoryMode.EVIDENCE_BOUND_RECONSTRUCTION:
                base.add("reconstruct_memory")
            return frozenset(base)
        allowed = self.state.task.available_memory_ids
        memory_ids = set(allowed) if allowed is not None else set(self.memories)
        if memory_ids - self.state.inspected_memory_ids:
            base.add("inspect_memory")
        decided = {guidance.memory_id for guidance in self.state.guidance_history}
        if (
            self.memory_mode is MemoryMode.EVIDENCE_BOUND_RECONSTRUCTION
            and self.state.inspected_memory_ids - decided
        ):
            base.add("reconstruct_memory")
        if self.policy_context_version == "v3" and hasattr(self, "state"):
            base = {
                name
                for name in base
                if self.skills[name].cost <= self.state.remaining_budget
            }
        return frozenset(base)

    def expose_memory(self, memory_id: str) -> Observation:
        """Mode-specific memory observation; all modes keep memory out of evidence."""
        allowed = self.state.task.available_memory_ids
        if allowed is not None and memory_id not in allowed:
            return Observation(False, "memory_context", error="memory is unavailable in this task")
        memory = self.memories.get(memory_id)
        if memory is None:
            return Observation(False, "memory_context", error="unknown strategy memory")
        self.state.inspected_memory_ids.add(memory_id)
        source = memory.source_state
        if self.memory_mode is MemoryMode.QUERY_GUIDANCE:
            guidance = Guidance(GuidanceKind.QUERY, memory.instruction, memory_id)
            self.state.guidance_history.append(guidance)
            return Observation(
                True,
                "memory_guidance",
                {
                    "memory_id": memory_id,
                    "kind": GuidanceKind.QUERY.value,
                    "instruction": memory.instruction,
                },
            )
        payload = {
            "memory_id": memory_id,
            "source_state": {
                "claim_facets": list(source.claim_facets),
                "corpus_scope": source.corpus_scope,
                "retrieved_document_types": list(source.retrieved_document_types),
                "unresolved_gaps": list(source.unresolved_gaps),
                "contradiction_state": source.contradiction_state,
            },
            "historical_strategy": {
                "kind": memory.guidance_kind.value,
                "instruction": memory.instruction,
            },
        }
        if self.memory_mode is MemoryMode.RAW_REPLAY:
            self.state.guidance_history.append(
                Guidance(memory.guidance_kind, memory.instruction, memory_id)
            )
        return Observation(True, "memory_context", payload)
