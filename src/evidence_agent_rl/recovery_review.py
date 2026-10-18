"""Validate independent AI reviews and build an auditable recovery SFT mix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .experiment import sha256_file
from .sft import SFTExample, validate_sft_examples


REVIEW_CHECKS = (
    "action_sequence_correct",
    "memory_boundary_correct",
    "evidence_chain_correct",
    "answer_correct",
    "train_split_safe",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"JSONL file is empty: {path}")
    return rows


def _jsonl_content_sha256(rows: Iterable[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_sft_example(raw: dict[str, Any]) -> SFTExample:
    return SFTExample(
        example_id=str(raw["example_id"]),
        task_id=str(raw["task_id"]),
        split=str(raw["split"]),
        source_paper_ids=tuple(raw["source_paper_ids"]),
        review_status=str(raw["review_status"]),
        step_index=int(raw["step_index"]),
        messages=tuple(raw["messages"]),
    )


def build_reviewed_recovery_mix(
    base_rows: Iterable[dict[str, Any]],
    candidates: Iterable[dict[str, Any]],
    reviews: Iterable[dict[str, Any]],
    *,
    minimum_ai_reviewers: int = 2,
    required_review_checks: Iterable[str] = REVIEW_CHECKS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Append unanimously approved draft recovery examples to a train SFT set."""

    if minimum_ai_reviewers < 2:
        raise ValueError("recovery promotion requires at least two AI reviewers")
    base = list(base_rows)
    candidate_rows = list(candidates)
    review_rows = list(reviews)
    frozen_review_checks = tuple(required_review_checks)
    if not frozen_review_checks or len(set(frozen_review_checks)) != len(
        frozen_review_checks
    ):
        raise ValueError("recovery review checks must be unique and non-empty")
    reviews_by_candidate: dict[str, list[dict[str, Any]]] = {}
    seen_reviews: set[tuple[str, str]] = set()
    for review in review_rows:
        if review.get("reviewer_kind") != "ai_model":
            raise ValueError("recovery review must declare reviewer_kind=ai_model")
        key = (str(review.get("candidate_id", "")), str(review.get("reviewer_id", "")))
        if not all(key):
            raise ValueError("recovery review requires candidate_id and reviewer_id")
        if key in seen_reviews:
            raise ValueError(f"duplicate recovery review: {key}")
        seen_reviews.add(key)
        reviews_by_candidate.setdefault(key[0], []).append(review)

    accepted_ids: list[str] = []
    accepted_candidates: dict[str, dict[str, Any]] = {}
    reviewer_ids: set[str] = set()
    recovery_examples: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        candidate_id = str(candidate["candidate_id"])
        if candidate.get("review_status") != "draft_requires_independent_review":
            raise ValueError(f"{candidate_id}: unexpected candidate review status")
        examples = list(candidate.get("examples", []))
        expected_ids = {str(example["example_id"]) for example in examples}
        approvals = reviews_by_candidate.get(candidate_id, [])
        valid_approvals = []
        for review in approvals:
            approved_ids = set(review.get("approved_example_ids", []))
            if (
                review.get("verdict") == "accept"
                and all(review.get(check) is True for check in frozen_review_checks)
                and approved_ids == expected_ids
            ):
                valid_approvals.append(review)
        distinct_reviewers = {str(review["reviewer_id"]) for review in valid_approvals}
        if len(distinct_reviewers) < minimum_ai_reviewers:
            continue
        accepted_ids.append(candidate_id)
        accepted_candidates[candidate_id] = candidate
        reviewer_ids.update(distinct_reviewers)

    source_task_ids = {
        str(candidate["source_task_id"])
        for candidate in accepted_candidates.values()
    }
    base_task_ids = {str(row["task_id"]) for row in base}
    missing_source_tasks = sorted(source_task_ids - base_task_ids)
    if missing_source_tasks:
        raise ValueError(
            "approved recovery candidates reference tasks absent from base SFT: "
            + ", ".join(missing_source_tasks)
        )
    replaced_base = [row for row in base if str(row["task_id"]) not in source_task_ids]
    for candidate_id in accepted_ids:
        candidate = accepted_candidates[candidate_id]
        source_task_id = str(candidate["source_task_id"])
        for example in candidate["examples"]:
            promoted = dict(example)
            promoted["task_id"] = source_task_id
            promoted["review_status"] = "draft"
            recovery_examples.append(promoted)

    if not recovery_examples:
        raise ValueError("no recovery candidate passed the AI review gate")
    combined = replaced_base + recovery_examples
    if len({row["example_id"] for row in combined}) != len(combined):
        raise ValueError("base and recovery SFT data contain duplicate example_id values")
    validate_sft_examples(_as_sft_example(row) for row in combined)
    report = {
        "version": 1,
        "epistemic_status": "dual_ai_reviewed_recovery_not_human_review",
        "minimum_ai_reviewers": minimum_ai_reviewers,
        "required_review_checks": list(frozen_review_checks),
        "reviewer_ids": sorted(reviewer_ids),
        "candidate_count": len(candidate_rows),
        "accepted_candidate_count": len(accepted_ids),
        "accepted_candidate_ids": accepted_ids,
        "base_examples": len(base),
        "replaced_base_examples": len(base) - len(replaced_base),
        "recovery_examples": len(recovery_examples),
        "total_examples": len(combined),
        "output_review_status": "draft",
    }
    return combined, report


def build_reviewed_recovery_mix_from_files(
    *,
    base_path: Path,
    candidate_path: Path,
    review_paths: Iterable[Path],
    output_path: Path,
    manifest_path: Path,
    minimum_ai_reviewers: int = 2,
    required_review_checks: Iterable[str] = REVIEW_CHECKS,
) -> dict[str, Any]:
    review_files = tuple(review_paths)
    base_rows = _read_jsonl(base_path)
    candidate_rows = _read_jsonl(candidate_path)
    reviews_by_file = {path: _read_jsonl(path) for path in review_files}
    reviews = [row for path in review_files for row in reviews_by_file[path]]
    combined, report = build_reviewed_recovery_mix(
        base_rows,
        candidate_rows,
        reviews,
        minimum_ai_reviewers=minimum_ai_reviewers,
        required_review_checks=required_review_checks,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in combined
            )
        )
    report.update(
        {
            "base_sha256": _jsonl_content_sha256(base_rows),
            "candidates_sha256": _jsonl_content_sha256(candidate_rows),
            "review_sha256": {
                path.as_posix(): _jsonl_content_sha256(reviews_by_file[path])
                for path in review_files
            },
            "output": output_path.as_posix(),
            "output_sha256": sha256_file(output_path),
        }
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest_file:
        manifest_file.write(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    return report
