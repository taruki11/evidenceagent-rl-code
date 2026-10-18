"""Machine-readable readiness gate for the priority double-review set."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .curation import load_page_records, validate_dataset_quality_gate
from .experiment import sha256_file
from .io import load_documents, load_tasks


@dataclass(frozen=True)
class ReviewReadinessReport:
    version: int
    ready: bool
    epistemic_status: str
    target: int
    selected_task_ids: tuple[str, ...]
    verified_tasks: int
    by_split: dict[str, int]
    blockers: tuple[str, ...]
    artifacts: dict[str, dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["selected_task_ids"] = list(self.selected_task_ids)
        result["blockers"] = list(self.blockers)
        return result


def build_double_review_readiness(
    *,
    selection_path: str | Path,
    reviewed_tasks_path: str | Path,
    reviewed_documents_path: str | Path,
    pages_path: str | Path,
) -> ReviewReadinessReport:
    selection_file = Path(selection_path).resolve()
    task_file = Path(reviewed_tasks_path).resolve()
    document_file = Path(reviewed_documents_path).resolve()
    pages_file = Path(pages_path).resolve()
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    if not isinstance(selection, dict):
        raise ValueError("review selection must be an object")
    raw_ids = selection.get("task_ids")
    target = selection.get("target")
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or any(not isinstance(item, str) or not item for item in raw_ids)
        or len(set(raw_ids)) != len(raw_ids)
        or not isinstance(target, int)
        or target != len(raw_ids)
    ):
        raise ValueError("review selection task IDs/target are invalid")
    selected_ids = tuple(raw_ids)
    artifacts: dict[str, dict[str, Any]] = {
        "selection": {
            "path": str(selection_file),
            "sha256": sha256_file(selection_file),
            "bytes": selection_file.stat().st_size,
        }
    }
    blockers: list[str] = []
    for name, path in (
        ("reviewed_tasks", task_file),
        ("reviewed_documents", document_file),
        ("pages", pages_file),
    ):
        if not path.is_file():
            blockers.append(f"required artifact is missing: {name}")
        else:
            artifacts[name] = {
                "path": str(path), "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    verified = 0
    by_split: dict[str, int] = {}
    if not blockers:
        try:
            documents = load_documents(document_file)
            task_map = {task.task_id: task for task in load_tasks(task_file)}
            missing = sorted(set(selected_ids) - set(task_map))
            if missing:
                blockers.append(f"reviewed task artifact lacks selected tasks: {missing}")
            else:
                selected = tuple(task_map[task_id] for task_id in selected_ids)
                pages = load_page_records(pages_file)
                for split in sorted({task.split for task in selected}):
                    split_tasks = tuple(task for task in selected if task.split == split)
                    validate_dataset_quality_gate(
                        split_tasks, documents, pages, gate="double_review", split=split
                    )
                    by_split[split] = len(split_tasks)
                verified = len(selected)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"double-review validation failed: {exc}")
    ready = not blockers and verified == target
    return ReviewReadinessReport(
        version=1, ready=ready,
        epistemic_status=(
            "double_review_artifacts_validated" if ready
            else "double_review_not_ready_no_human_claim"
        ),
        target=target, selected_task_ids=selected_ids, verified_tasks=verified,
        by_split=by_split, blockers=tuple(blockers), artifacts=artifacts,
    )


def validate_double_review_readiness(
    raw: object, *, repo_root: str | Path
) -> ReviewReadinessReport:
    """Rebuild a ready report from its bound human-review artifacts."""

    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("double-review readiness report must be version 1")
    if raw.get("ready") is not True or raw.get("blockers") != []:
        raise ValueError("double-review readiness report is not ready")
    if raw.get("epistemic_status") != "double_review_artifacts_validated":
        raise ValueError("double-review readiness epistemic status is invalid")
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("double-review readiness report lacks artifacts")
    root = Path(repo_root).resolve()
    paths: dict[str, Path] = {}
    for name in ("selection", "reviewed_tasks", "reviewed_documents", "pages"):
        entry = artifacts.get(name)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"double-review readiness lacks artifact: {name}")
        path = Path(entry["path"]).resolve()
        if root != path and root not in path.parents:
            raise ValueError(f"double-review artifact escapes repository: {name}")
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"double-review artifact hash mismatch: {name}")
        if path.stat().st_size != entry.get("bytes"):
            raise ValueError(f"double-review artifact size mismatch: {name}")
        paths[name] = path
    rebuilt = build_double_review_readiness(
        selection_path=paths["selection"],
        reviewed_tasks_path=paths["reviewed_tasks"],
        reviewed_documents_path=paths["reviewed_documents"],
        pages_path=paths["pages"],
    )
    if rebuilt.to_json() != raw:
        raise ValueError("double-review readiness report drift")
    return rebuilt
