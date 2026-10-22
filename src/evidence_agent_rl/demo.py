"""Offline, scripted evidence lifecycle demo using only synthetic records.

This exercises harness bookkeeping and deterministic checks, not model reasoning
or semantic entailment. Run ``python -m evidence_agent_rl.demo`` after installing
the package, or set PYTHONPATH to ``src`` in a source checkout.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .harness import EvidenceHarness
from .models import Action, Document, Span, TaskSpec
from .verifier import DeterministicVerifier


def run_demo() -> dict:
    """Return a reproducible report without files, network, or provider calls."""
    # These identifiers are deliberately synthetic, not real publications/DOIs.
    document = Document(
        document_id="synthetic-note",
        title="Synthetic lantern measurement",
        doi="synthetic:lantern-note",
        corpus_scope="synthetic-demo",
        document_type="synthetic_note",
        spans=(Span("measurement", "The synthetic lantern glows blue.", ("color",)),),
    )
    task = TaskSpec(
        task_id="synthetic-color",
        question="What color does the synthetic lantern glow?",
        corpus_scope="synthetic-demo",
        required_facets=("color",),
        allowed_dois=frozenset({document.doi}),
        deterministic_answer_key={"color": ("blue",)},
    )
    harness = EvidenceHarness([document])
    harness.reset(task, episode_id="accepted-demo")
    for action in (
        Action("search", {"query": "synthetic lantern"}),
        Action("retrieve", {"document_id": document.document_id}),
        Action(
            "inspect_span",
            {"document_id": document.document_id, "span_id": "measurement"},
        ),
    ):
        observation = harness.step(action)
        if not observation.ok:
            raise RuntimeError(f"Demo action failed: {observation.error}")
    handle_id = observation.data["handle_id"]
    for action in (
        Action("check_citation", {"handle_id": handle_id}),
        Action(
            "answer",
            {
                "claims": [
                    {"text": "The lantern glows blue.", "evidence_handles": [handle_id]}
                ]
            },
        ),
    ):
        observation = harness.step(action)
        if not observation.ok:
            raise RuntimeError(f"Demo action failed: {observation.error}")
    verifier = DeterministicVerifier()
    score = verifier.score(harness.state)
    accepted_trace = [asdict(step) for step in harness.state.trace]
    replay = EvidenceHarness([document])
    replay.replay(task, harness.cloned_action_trace(), episode_id="accepted-demo")
    replay_consistent = (
        replay.state == harness.state and verifier.score(replay.state) == score
    )

    # A valid handle from another episode must not become evidence here.
    rejected = EvidenceHarness([document])
    rejected.reset(task, episode_id="rejected-demo")
    rejection = rejected.step(
        Action(
            "answer",
            {
                "claims": [
                    {"text": "The lantern glows blue.", "evidence_handles": [handle_id]}
                ],
            },
        )
    )
    rejected_score = verifier.score(rejected.state)
    blocked = (
        not rejection.ok
        and rejected.state.firewall_violations == 1
        and not rejected.state.final_claims
    )
    return {
        "schema_version": 1,
        "mode": "offline_scripted_synthetic",
        "limitations": [
            "No model inference or training is performed.",
            "Deterministic token and provenance checks do not establish semantic entailment.",
            "This single synthetic example is not a benchmark or evidence of real-world reliability.",
        ],
        "accepted": {
            "success": verifier.is_success(score),
            "score": asdict(score),
            "total_reward": score.total,
            "claims": [asdict(claim) for claim in harness.state.final_claims],
            "remaining_budget": harness.state.remaining_budget,
            "trace": accepted_trace,
        },
        "rejected": {
            "blocked": blocked,
            "reason": rejection.error,
            "success": verifier.is_success(rejected_score),
            "trace": [asdict(step) for step in rejected.state.trace],
        },
        "replay_consistent": replay_consistent,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Also save JSON to a new file; existing files are never overwritten.",
    )
    args = parser.parse_args(argv)
    report = run_demo()
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        except OSError as exc:
            parser.error(f"Cannot create output file: {exc}")
    print(payload, end="")
    return (
        0
        if report["accepted"]["success"]
        and report["rejected"]["blocked"]
        and report["replay_consistent"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
