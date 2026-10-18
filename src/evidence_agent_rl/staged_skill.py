"""Candidate C: a two-stage instruction for the HotpotQA line.

Stage SELECT shows the original *selection* guidance (no short-answer constraint).  Stage ANSWER
shows the minimal-answer guidance.  The two texts are loaded as ordinary policy-skill contracts, so
each has its own SHA-256 and both are recorded in the run identity and in each decision's context.

DECLARED FIDELITY LIMIT (measured before running; see
`docs/HOTPOTQA_STAGE_C_MECHANISM_20260929.md`):

    The stated mechanism is "constrain the short answer only at the final answer decision".  On this
    harness that decision is NOT identifiable from public state in advance: across arms A, B and the
    privileged teacher, 97.4%-99.2% of answer steps were taken while a further search+select pair was
    still affordable and selectable spans remained.  Any boundary that waits for gathering to finish
    therefore almost never fires; any boundary that fires in time fires while gathering is still in
    progress.

    Boundary S below is the closest implementable rule.  It preserves the original selection guidance
    for the pure-gathering steps (no checked handle yet) and switches to the answer guidance from the
    first step on which the answer decision becomes live.  It is therefore a WEAKER proxy for the
    stated mechanism, and a negative result bounds the proxy, not the mechanism.

Nothing here reads gold, decides selection, or fills a supporting set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .policy_skill import PolicySkillContract, load_policy_skill_contract

STAGE_SELECTION = "selection"
STAGE_ANSWER = "answer"

_SKILL_ACTIONS = ("search", "select_evidence", "answer", "abstain")


@dataclass(frozen=True)
class StagedSkill:
    """Two policy-skill contracts plus the declared boundary between them."""

    selection: PolicySkillContract
    answer: PolicySkillContract
    boundary: str = "answer_available_and_last_step_added_evidence"

    def identity(self) -> dict[str, Any]:
        """The identity recorded in the run; both stages are always named with their SHA-256."""
        return {
            "candidate": "staged_two_phase",
            "boundary": self.boundary,
            "stage_selection": {"name": self.selection.name, "sha256": self.selection.sha256},
            "stage_answer": {"name": self.answer.name, "sha256": self.answer.sha256},
        }

    def stage_for(self, *, answer_available: bool, last_step_added_evidence: bool) -> str:
        """Boundary S, as declared: both conditions must hold for the answer stage."""
        if answer_available and last_step_added_evidence:
            return STAGE_ANSWER
        return STAGE_SELECTION

    def contract_for(self, stage: str) -> PolicySkillContract:
        return self.answer if stage == STAGE_ANSWER else self.selection


def load_staged_skill(selection_path: str | Any, answer_path: str | Any) -> StagedSkill:
    selection = load_policy_skill_contract(selection_path, required_actions=_SKILL_ACTIONS)
    answer = load_policy_skill_contract(answer_path, required_actions=_SKILL_ACTIONS)
    if selection.sha256 == answer.sha256:
        raise ValueError("the two stages must be different instruction texts")
    return StagedSkill(selection=selection, answer=answer)


@dataclass
class StagedInstructionSelector:
    """Callable passed to `ReActRunner` in place of a fixed contract.

    It tracks the episode's checked-handle count between decisions, so the boundary uses only public
    state.  Every decision's stage is appended to `stages` for the trajectory record.
    """

    staged: StagedSkill
    _previous_checked: int | None = None
    _previous_episode: str | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, harness: Any) -> PolicySkillContract:
        checked = 0
        episode = None
        if hasattr(harness, "state"):
            checked = sum(1 for h in harness.state.evidence.values() if h.checked)
            episode = harness.state.episode_id
        if episode != self._previous_episode:
            self._previous_checked = None
            self._previous_episode = episode
        grew = self._previous_checked is None or checked > self._previous_checked
        self._previous_checked = checked
        try:
            answer_available = "answer" in harness._available_skill_names()
        except Exception:
            answer_available = False
        stage = self.staged.stage_for(answer_available=answer_available,
                                      last_step_added_evidence=grew)
        self.stages.append({"stage": stage, "checked": checked, "answer_available": answer_available,
                            "last_step_added_evidence": grew,
                            "instructions_sha256": self.staged.contract_for(stage).sha256})
        return self.staged.contract_for(stage)

    def stage_sequence(self) -> list[str]:
        return [s["stage"] for s in self.stages]
