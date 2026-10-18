"""Deterministic checks and transparent reward accounting."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from .models import EpisodeState


@dataclass(frozen=True)
class RewardBreakdown:
    task: float
    evidence: float
    citation: float
    abstain: float
    format: float
    tool_cost: float
    unsupported: float
    memory_violation: float
    token_cost: float = 0.0

    @property
    def total(self) -> float:
        return (
            3 * self.task
            + 2 * self.evidence
            + self.citation
            + self.abstain
            + 0.5 * self.format
            - self.tool_cost
            - self.unsupported
            - self.memory_violation
            - self.token_cost
        )


class DeterministicVerifier:
    """First-layer verifier; semantic entailment is intentionally out of scope."""

    @staticmethod
    def is_success(score: RewardBreakdown) -> bool:
        """Deployment gate, stricter than the answer-correctness component alone."""
        return (
            score.task == 1.0
            and score.unsupported == 0.0
            and score.memory_violation == 0.0
            and score.format == 1.0
        )

    def score(self, state: EpisodeState, *, token_count: int = 0) -> RewardBreakdown:
        if token_count < 0:
            raise ValueError("token_count cannot be negative")
        used_tools = sum(step.action.name not in {"answer", "abstain"} for step in state.trace)
        valid_format = not any(step.observation.kind == "invalid_action" for step in state.trace)
        mismatch_acceptance = float(
            state.task.mismatch_role == "mismatched"
            and any(guidance.kind.value != "<EMPTY>" for guidance in state.guidance_history)
        )
        memory_violations = float(state.firewall_violations) + mismatch_acceptance
        if state.abstention_reason is not None:
            correct_abstention = float(state.task.insufficient)
            return RewardBreakdown(
                task=correct_abstention,
                evidence=0.0,
                citation=0.0,
                abstain=1.0 if state.task.insufficient else -1.0,
                format=float(valid_format),
                tool_cost=0.15 * used_tools,
                unsupported=0.0,
                memory_violation=memory_violations,
                token_cost=0.00002 * token_count,
            )

        required = set(state.task.required_facets)
        supported_claims = 0
        cited_handles: set[str] = set()
        evidence_covered_facets: set[str] = set()
        for claim in state.final_claims:
            handles = [state.evidence.get(item) for item in claim.evidence_handles]
            matching_facets = {
                facet
                for facet in required
                if state.task.deterministic_answer_key.get(facet)
                and all(
                    _matches_token(token, claim.text)
                    for token in state.task.deterministic_answer_key[facet]
                )
            }
            evidence_facets = {
                facet
                for handle in handles
                if handle is not None and handle.checked
                for facet in handle.facets
            }
            evidence_covered_facets.update(evidence_facets)
            if handles and matching_facets and matching_facets <= evidence_facets:
                supported_claims += 1
            cited_handles.update(claim.evidence_handles)

        claim_count = len(state.final_claims)
        coverage = len(required & evidence_covered_facets) / len(required) if required else 1.0
        citation = (
            sum(
                handle_id in state.evidence and state.evidence[handle_id].checked
                for handle_id in cited_handles
            )
            / len(cited_handles)
            if cited_handles
            else 0.0
        )
        unsupported_count = claim_count - supported_claims
        answer_key_matches = True
        for facet in required:
            required_tokens = state.task.deterministic_answer_key.get(facet)
            matching_claims = [
                claim
                for claim in state.final_claims
                if required_tokens
                and all(_matches_token(token, claim.text) for token in required_tokens)
            ]
            if not required_tokens or not matching_claims:
                answer_key_matches = False
                continue
            if not any(
                all(_matches_token(token, claim.text) for token in required_tokens)
                for claim in matching_claims
            ):
                answer_key_matches = False
        if any(
            _matches_token(token, claim.text)
            for token in state.task.forbidden_answer_tokens
            for claim in state.final_claims
        ):
            answer_key_matches = False
        query_matches = all(
            any(_matches_token(term, query) for query in state.query_history)
            for term in state.task.required_query_terms
        )
        task_success = float(
            state.done
            and not state.task.insufficient
            and coverage == 1.0
            and unsupported_count == 0
            and answer_key_matches
            and query_matches
        )
        return RewardBreakdown(
            task=task_success,
            evidence=coverage,
            citation=citation,
            abstain=-1.0 if state.task.insufficient and claim_count else 0.0,
            format=float(state.done and claim_count > 0 and valid_format),
            tool_cost=0.15 * used_tools,
            unsupported=float(unsupported_count),
            memory_violation=memory_violations,
            token_cost=0.00002 * token_count,
        )


def _normalize_answer_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.translate(str.maketrans({"\u2212": "-", "\u2013": "-", "\u2014": "-"}))
    return re.sub(r"(?<=\d),(?=\d)", "", normalized)


def _matches_token(token: str, text: str) -> bool:
    normalized_token = _normalize_answer_text(token).strip()
    if not normalized_token:
        return False
    return re.search(
        rf"(?<!\w){re.escape(normalized_token)}(?!\w)", _normalize_answer_text(text)
    ) is not None
