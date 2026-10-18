"""Dependency-free audit and launch planning for Harness-derived TRL SFT."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .experiment import sha256_file
from .react import JSONActionCodec, POLICY_SYSTEM_PROMPT, serialize_policy_context
from .sft import SFTExample, validate_public_context, validate_sft_examples
from .policy_skill import POLICY_SKILL_NAME, validate_policy_skill_binding


TRAINING_PACKAGES = ("torch", "transformers", "datasets", "trl", "peft", "accelerate", "trackio")


def validate_resume_checkpoint(path: Path) -> Path:
    """Validate that a checkpoint contains the state needed for exact SFT resume."""
    checkpoint = path.resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"resume checkpoint directory does not exist: {checkpoint}")
    required = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(
            f"resume checkpoint is incomplete: {checkpoint}; missing {missing}"
        )
    return checkpoint


def latest_valid_sft_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the highest-step complete Trainer checkpoint, if any."""

    root = Path(output_dir).resolve()
    if not root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        raw_step = path.name.removeprefix("checkpoint-")
        if raw_step.isdigit():
            candidates.append((int(raw_step), path))
    for _, path in sorted(candidates, reverse=True):
        try:
            return validate_resume_checkpoint(path)
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class SFTDatasetAudit:
    path: str
    sha256: str
    bytes: int
    examples: int
    tasks: int
    source_papers: int
    splits: dict[str, int]
    review_statuses: dict[str, int]
    actions: dict[str, int]
    policy_skill_name: str
    policy_skill_sha256: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SFTLaunchPlan:
    experiment_id: str
    config_sha256: str
    launch_requested: bool
    preflight_ready: bool
    dataset: SFTDatasetAudit
    validation_dataset: SFTDatasetAudit | None
    model: dict[str, str]
    output_dir: str
    training: dict[str, Any]
    dependency_versions: dict[str, str | None]
    curriculum: dict[str, Any]
    initial_adapter: dict[str, str] | None

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["dataset"] = self.dataset.to_json()
        result["validation_dataset"] = (
            self.validation_dataset.to_json() if self.validation_dataset else None
        )
        return result


def apply_memory_trajectory_curriculum(
    records: Iterable[dict[str, Any]],
    *,
    repeat_factor: int,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Deterministically oversample complete memory trajectories.

    This changes sampling frequency only. It does not create new actions, factual
    labels, or evidence. Synthetic identifiers keep repeated records traceable.
    """

    if not isinstance(repeat_factor, int) or isinstance(repeat_factor, bool):
        raise ValueError("memory trajectory repeat factor must be an integer")
    if not 1 <= repeat_factor <= 16:
        raise ValueError("memory trajectory repeat factor must be in [1, 16]")
    base = tuple(dict(record) for record in records)
    if not base:
        raise ValueError("memory trajectory curriculum requires records")

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    actions_by_task: dict[str, set[str]] = defaultdict(set)
    base_action_counts: Counter[str] = Counter()
    for record in base:
        task_id = str(record["task_id"])
        action = JSONActionCodec.decode(record["completion"][0]["content"])
        by_task[task_id].append(record)
        actions_by_task[task_id].add(action.name)
        base_action_counts[action.name] += 1

    memory_tasks = sorted(
        task_id
        for task_id, actions in actions_by_task.items()
        if actions & {"inspect_memory", "reconstruct_memory"}
    )
    if not memory_tasks:
        if repeat_factor != 1:
            raise ValueError("memory trajectory curriculum found no memory tasks")
        return base, {
            "epistemic_status": "no_oversampling_requested",
            "repeat_factor": 1,
            "base_examples": len(base),
            "effective_examples": len(base),
            "memory_tasks": 0,
            "matched_memory_tasks": 0,
            "mismatched_memory_tasks": 0,
            "base_action_counts": dict(sorted(base_action_counts.items())),
            "effective_action_counts": dict(sorted(base_action_counts.items())),
            "base_memory_action_rate": 0.0,
            "effective_memory_action_rate": 0.0,
        }
    matched = [task_id for task_id in memory_tasks if task_id.endswith("-matched")]
    mismatched = [task_id for task_id in memory_tasks if task_id.endswith("-mismatched")]
    if repeat_factor == 1:
        memory_actions = {"inspect_memory", "reconstruct_memory"}
        memory_action_count = sum(base_action_counts[name] for name in memory_actions)
        return base, {
            "epistemic_status": "no_oversampling_requested",
            "repeat_factor": 1,
            "base_examples": len(base),
            "effective_examples": len(base),
            "memory_tasks": len(memory_tasks),
            "matched_memory_tasks": len(matched),
            "mismatched_memory_tasks": len(mismatched),
            "base_action_counts": dict(sorted(base_action_counts.items())),
            "effective_action_counts": dict(sorted(base_action_counts.items())),
            "base_memory_action_rate": memory_action_count / len(base),
            "effective_memory_action_rate": memory_action_count / len(base),
        }
    unlabelled = sorted(set(memory_tasks) - set(matched) - set(mismatched))
    if unlabelled:
        raise ValueError(f"memory tasks lack matched/mismatched suffixes: {unlabelled}")
    matched_stems = {task_id[: -len("-matched")] for task_id in matched}
    mismatched_stems = {
        task_id[: -len("-mismatched")] for task_id in mismatched
    }
    if matched_stems != mismatched_stems:
        raise ValueError(
            "memory trajectory curriculum requires balanced matched/mismatched tasks"
        )

    expanded = list(base)
    for replica in range(1, repeat_factor):
        suffix = f"::memory-repeat-{replica:02d}"
        for task_id in memory_tasks:
            for record in by_task[task_id]:
                clone = dict(record)
                clone["example_id"] = str(record["example_id"]) + suffix
                clone["task_id"] = task_id + suffix
                expanded.append(clone)

    effective_action_counts: Counter[str] = Counter()
    for record in expanded:
        action = JSONActionCodec.decode(record["completion"][0]["content"])
        effective_action_counts[action.name] += 1
    memory_actions = {"inspect_memory", "reconstruct_memory"}
    base_memory_actions = sum(base_action_counts[name] for name in memory_actions)
    effective_memory_actions = sum(
        effective_action_counts[name] for name in memory_actions
    )
    report = {
        "epistemic_status": "deterministic_oversampling_of_existing_sft_not_new_labels",
        "repeat_factor": repeat_factor,
        "base_examples": len(base),
        "effective_examples": len(expanded),
        "memory_tasks": len(memory_tasks),
        "matched_memory_tasks": len(matched),
        "mismatched_memory_tasks": len(mismatched),
        "base_action_counts": dict(sorted(base_action_counts.items())),
        "effective_action_counts": dict(sorted(effective_action_counts.items())),
        "base_memory_action_rate": base_memory_actions / len(base),
        "effective_memory_action_rate": effective_memory_actions / len(expanded),
    }
    return tuple(expanded), report


def latest_validation_metrics(log_history: Iterable[object]) -> dict[str, Any] | None:
    """Return the latest in-training validation record without reopening trackers."""

    for entry in reversed(tuple(log_history)):
        if isinstance(entry, dict) and "eval_loss" in entry:
            return dict(entry)
    return None


def apply_memory_decision_curriculum(
    records: Iterable[dict[str, Any]],
    *,
    repeat_factor: int,
    include_base_records: bool = False,
    task_pair_roles: dict[str, tuple[str, str]] | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Oversample paired memory decisions, optionally retaining all base anchors."""

    if not isinstance(repeat_factor, int) or isinstance(repeat_factor, bool):
        raise ValueError("memory decision repeat factor must be an integer")
    if not 1 <= repeat_factor <= 16:
        raise ValueError("memory decision repeat factor must be in [1, 16]")
    if not isinstance(include_base_records, bool):
        raise ValueError("memory decision include-base flag must be boolean")
    base = tuple(dict(record) for record in records)
    selected: list[dict[str, Any]] = []
    actions_by_task: dict[str, set[str]] = defaultdict(set)
    for record in base:
        action = JSONActionCodec.decode(record["completion"][0]["content"])
        if action.name not in {"inspect_memory", "reconstruct_memory"}:
            continue
        selected.append(record)
        actions_by_task[str(record["task_id"])].add(action.name)
    if not selected:
        raise ValueError("memory decision curriculum found no decision examples")
    incomplete = sorted(
        task_id
        for task_id, actions in actions_by_task.items()
        if actions != {"inspect_memory", "reconstruct_memory"}
    )
    if incomplete:
        raise ValueError(f"memory decision tasks lack complete action pairs: {incomplete}")
    if task_pair_roles is None:
        pair_roles = {
            task_id: (task_id[: -len("-matched")], "matched")
            for task_id in actions_by_task
            if task_id.endswith("-matched")
        }
        pair_roles.update({
            task_id: (task_id[: -len("-mismatched")], "mismatched")
            for task_id in actions_by_task
            if task_id.endswith("-mismatched")
        })
    else:
        pair_roles = {
            task_id: task_pair_roles[task_id]
            for task_id in actions_by_task
            if task_id in task_pair_roles
        }
    roles_by_pair: dict[str, set[str]] = defaultdict(set)
    for pair_id, role in pair_roles.values():
        roles_by_pair[pair_id].add(role)
    if (
        set(pair_roles) != set(actions_by_task)
        or any(roles != {"matched", "mismatched"} for roles in roles_by_pair.values())
        or len(actions_by_task) != 2 * len(roles_by_pair)
    ):
        raise ValueError(
            "memory decision curriculum requires paired matched/mismatched tasks"
        )
    expanded: list[dict[str, Any]] = list(base) if include_base_records else []
    first_replica = 1 if include_base_records else 0
    for replica in range(first_replica, repeat_factor):
        suffix = "" if replica == 0 else f"::decision-repeat-{replica:02d}"
        for record in selected:
            clone = dict(record)
            clone["example_id"] = str(record["example_id"]) + suffix
            clone["task_id"] = str(record["task_id"]) + suffix
            expanded.append(clone)
    counts: Counter[str] = Counter(
        JSONActionCodec.decode(record["completion"][0]["content"]).name
        for record in expanded
    )
    return tuple(expanded), {
        "epistemic_status": (
            "anchor_preserving_decision_oversampling_of_existing_sft_not_new_labels"
            if include_base_records
            else "decision_focused_oversampling_of_existing_sft_not_new_labels"
        ),
        "repeat_factor": repeat_factor,
        "include_base_records": include_base_records,
        "base_examples": len(base),
        "selected_examples": len(selected),
        "anchor_examples": len(base) - len(selected) if include_base_records else 0,
        "effective_examples": len(expanded),
        "memory_tasks": len(actions_by_task),
        "matched_memory_tasks": len(roles_by_pair),
        "mismatched_memory_tasks": len(roles_by_pair),
        "effective_action_counts": dict(sorted(counts.items())),
    }


def apply_public_transition_curriculum(
    records: Iterable[dict[str, Any]],
    *,
    repeat_factor: int,
    task_type_by_id: dict[str, str],
    task_id_prefix: str | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Oversample observed phase changes without creating new action labels."""

    if not isinstance(repeat_factor, int) or isinstance(repeat_factor, bool):
        raise ValueError("public transition repeat factor must be an integer")
    if not 2 <= repeat_factor <= 16:
        raise ValueError("public transition repeat factor must be in [2, 16]")
    base = tuple(dict(record) for record in records)
    if not base:
        raise ValueError("public transition curriculum requires records")

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in base:
        task_id = str(record["task_id"])
        if task_id not in task_type_by_id:
            raise ValueError(f"public transition task is absent from task bank: {task_id}")
        by_task[task_id].append(record)

    selected: list[dict[str, Any]] = []
    phase_counts: Counter[str] = Counter()
    phase_task_counts: dict[str, set[str]] = defaultdict(set)
    incomplete: dict[str, list[str]] = {}
    required_multihop = {
        "first_citation_check",
        "second_search",
        "second_citation_check",
        "terminal_answer",
    }
    for task_id, task_records in by_task.items():
        if task_id_prefix is not None and not task_id.startswith(task_id_prefix):
            continue
        task_type = task_type_by_id[task_id]
        if task_type not in {"multi_hop", "insufficient"}:
            continue
        actions = [
            (record, JSONActionCodec.decode(record["completion"][0]["content"]).name)
            for record in sorted(task_records, key=lambda item: str(item["example_id"]))
        ]
        observed_phases: set[str] = set()
        search_count = 0
        citation_count = 0
        selection_count = 0
        for record, action_name in actions:
            phase = None
            if task_type == "multi_hop":
                if action_name == "search":
                    search_count += 1
                    if search_count == 2:
                        phase = "second_search"
                elif action_name == "check_citation":
                    citation_count += 1
                    if citation_count == 1:
                        phase = "first_citation_check"
                    elif citation_count == 2:
                        phase = "second_citation_check"
                elif action_name == "select_evidence":
                    selection_count += 1
                    if selection_count == 1:
                        phase = "first_citation_check"
                    elif selection_count == 2:
                        phase = "second_citation_check"
                elif action_name == "answer":
                    phase = "terminal_answer"
            elif action_name == "abstain":
                phase = "terminal_abstain"
            if phase is None:
                continue
            selected.append(record)
            observed_phases.add(phase)
            phase_counts[phase] += 1
            phase_task_counts[phase].add(task_id)
        required = required_multihop if task_type == "multi_hop" else {"terminal_abstain"}
        missing = sorted(required - observed_phases)
        if missing:
            incomplete[task_id] = missing
    if incomplete:
        raise ValueError(f"public transition tasks lack required phases: {incomplete}")
    if not selected:
        raise ValueError("public transition curriculum found no transition examples")

    expanded = list(base)
    for replica in range(1, repeat_factor):
        suffix = f"::transition-repeat-{replica:02d}"
        for record in selected:
            clone = dict(record)
            clone["example_id"] = str(record["example_id"]) + suffix
            clone["task_id"] = str(record["task_id"]) + suffix
            expanded.append(clone)
    effective_actions: Counter[str] = Counter(
        JSONActionCodec.decode(record["completion"][0]["content"]).name
        for record in expanded
    )
    return tuple(expanded), {
        "epistemic_status": (
            "anchor_preserving_public_transition_oversampling_of_existing_sft_not_new_labels"
        ),
        "repeat_factor": repeat_factor,
        "task_id_prefix": task_id_prefix,
        "base_examples": len(base),
        "selected_examples": len(selected),
        "anchor_examples": len(base) - len(selected),
        "effective_examples": len(expanded),
        "phase_counts": dict(sorted(phase_counts.items())),
        "phase_task_counts": {
            phase: len(task_ids) for phase, task_ids in sorted(phase_task_counts.items())
        },
        "effective_action_counts": dict(sorted(effective_actions.items())),
    }


def audit_sft_token_lengths(
    records: Iterable[dict[str, Any]],
    tokenizer: Any,
    *,
    max_sequence_length: int,
) -> dict[str, Any]:
    """Measure rendered chat lengths across Transformers return-type versions."""

    lengths: list[int] = []
    overlength_example_ids: list[str] = []
    for record in records:
        token_ids = tokenizer.apply_chat_template(
            record["prompt"] + record["completion"],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
            **record.get("chat_template_kwargs", {}),
        )
        if not isinstance(token_ids, list) or any(
            not isinstance(token_id, int) for token_id in token_ids
        ):
            raise ValueError("chat template did not return a flat token-id list")
        length = len(token_ids)
        lengths.append(length)
        if length > max_sequence_length:
            overlength_example_ids.append(str(record["example_id"]))
    if not lengths:
        raise ValueError("token audit requires at least one SFT record")
    return {
        "examples": len(lengths),
        "minimum_tokens": min(lengths),
        "maximum_tokens": max(lengths),
        "max_sequence_length": max_sequence_length,
        "overlength_examples": len(overlength_example_ids),
        "overlength_example_ids": overlength_example_ids,
    }


def training_dependency_versions(
    packages: Iterable[str] = TRAINING_PACKAGES,
) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _load_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid SFT JSONL at line {line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"SFT row {line_number} must be an object")
        rows.append(row)
    if not rows:
        raise ValueError("SFT dataset must contain at least one example")
    return tuple(rows)


def _validate_schema_value(value: Any, schema: dict[str, Any], path: str) -> None:
    branches = schema.get("oneOf")
    if isinstance(branches, list):
        matches = 0
        for branch in branches:
            try:
                _validate_schema_value(value, branch, path)
            except ValueError:
                continue
            matches += 1
        if matches != 1:
            raise ValueError(f"{path} must match exactly one allowed schema branch")
        return
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} differs from the required constant")
    expected_type = schema.get("type")
    if expected_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"{path} is outside the allowed enum")
    elif expected_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")
        item_schema = schema.get("items", {})
        for index, item in enumerate(value):
            _validate_schema_value(item, item_schema, f"{path}[{index}]")
    elif expected_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        missing = required - set(value)
        if missing:
            raise ValueError(f"{path} is missing required fields: {sorted(missing)}")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise ValueError(f"{path} contains extra fields: {sorted(extra)}")
        for key, child in value.items():
            if key in properties:
                _validate_schema_value(child, properties[key], f"{path}.{key}")


def load_and_audit_sft_dataset(
    path: str | Path,
    *,
    expected_split: str,
    quality_gate: str,
    expected_tasks: int | None = None,
    expected_policy_context_version: str | None = None,
) -> tuple[SFTDatasetAudit, tuple[dict[str, Any], ...]]:
    """Validate exported actions and return TRL prompt-completion records."""

    dataset_path = Path(path).resolve()
    if not dataset_path.is_file():
        raise ValueError(f"SFT dataset is missing: {dataset_path}")
    rows = _load_rows(dataset_path)
    examples: list[SFTExample] = []
    task_steps: dict[str, list[int]] = defaultdict(list)
    action_counts: Counter[str] = Counter()
    policy_skills: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, 1):
        required = {
            "example_id",
            "task_id",
            "split",
            "source_paper_ids",
            "review_status",
            "step_index",
            "messages",
        }
        if set(row) != required:
            raise ValueError(
                f"SFT row {index} fields differ from the frozen schema: "
                f"missing={sorted(required - set(row))}, extra={sorted(set(row) - required)}"
            )
        for field in ("example_id", "task_id", "split", "review_status"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"SFT row {index} requires a non-empty string {field}")
        raw_messages = row["messages"]
        if not isinstance(raw_messages, list) or len(raw_messages) != 3:
            raise ValueError(f"SFT row {index} must contain exactly three messages")
        messages: list[dict[str, str]] = []
        for message in raw_messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError(f"SFT row {index} has an invalid message object")
            if not isinstance(message["role"], str) or not isinstance(message["content"], str):
                raise ValueError(f"SFT row {index} message values must be strings")
            messages.append(dict(message))
        if messages[0]["content"] != POLICY_SYSTEM_PROMPT:
            raise ValueError(f"SFT row {index} does not use the canonical policy prompt")
        try:
            context = json.loads(messages[1]["content"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"SFT row {index} contains invalid policy context JSON") from exc
        if not isinstance(context, dict):
            raise ValueError(f"SFT row {index} policy context must be an object")
        if expected_policy_context_version in {"v3", "v4", "v5", "v6"}:
            if context.get("policy_context_version") != expected_policy_context_version:
                raise ValueError(
                    f"SFT row {index} is not policy context {expected_policy_context_version}"
                )
        elif expected_policy_context_version in {"v1", "v2"}:
            if "policy_context_version" in context:
                raise ValueError(
                    f"SFT row {index} unexpectedly uses a versioned policy context"
                )
        validate_public_context(context)
        validate_policy_skill_binding(context)
        identity = context["skill_contract"]
        if identity["name"] != POLICY_SKILL_NAME:
            raise ValueError(
                f"SFT row {index} does not use the formal {POLICY_SKILL_NAME} policy skill"
            )
        policy_skills.add((identity["name"], identity["sha256"]))
        if serialize_policy_context(context) != messages[1]["content"]:
            raise ValueError(f"SFT row {index} policy context is not canonical")
        action = JSONActionCodec.decode(messages[2]["content"])
        raw_skills = context.get("skills")
        if not isinstance(raw_skills, list):
            raise ValueError(f"SFT row {index} policy context lacks a skill catalog")
        skill_schemas = {
            skill.get("name"): skill.get("parameters")
            for skill in raw_skills
            if isinstance(skill, dict)
            and isinstance(skill.get("name"), str)
            and isinstance(skill.get("parameters"), dict)
        }
        if action.name not in skill_schemas:
            raise ValueError(f"SFT row {index} action is absent from its public skill catalog")
        _validate_schema_value(
            action.arguments,
            skill_schemas[action.name],
            f"SFT row {index} action.arguments",
        )
        canonical_action = json.dumps(
            {"name": action.name, "arguments": action.arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if canonical_action != messages[2]["content"]:
            raise ValueError(f"SFT row {index} action is not canonical")
        source_papers = row["source_paper_ids"]
        if not isinstance(source_papers, list) or not source_papers or any(
            not isinstance(value, str) or not value for value in source_papers
        ):
            raise ValueError(f"SFT row {index} requires source_paper_ids")
        step_index = row["step_index"]
        if not isinstance(step_index, int) or isinstance(step_index, bool) or step_index < 0:
            raise ValueError(f"SFT row {index} has invalid step_index")
        example = SFTExample(
            example_id=row["example_id"],
            task_id=row["task_id"],
            split=row["split"],
            source_paper_ids=tuple(source_papers),
            review_status=row["review_status"],
            step_index=step_index,
            messages=tuple(messages),
        )
        examples.append(example)
        task_steps[example.task_id].append(step_index)
        action_counts[action.name] += 1
    validate_sft_examples(examples)
    if len(policy_skills) != 1:
        raise ValueError("SFT dataset mixes policy skill identities")
    policy_skill_name, policy_skill_sha256 = next(iter(policy_skills))
    if {example.split for example in examples} != {expected_split}:
        raise ValueError(f"SFT dataset must contain only split={expected_split}")
    if quality_gate in {
        "auto_provenance",
        "phase3_t1_correction_traces",
        "crosssource_correction_traces",
    }:
        allowed_statuses = {"draft", "single_review", "double_review"}
    elif quality_gate == "single_review":
        allowed_statuses = {"single_review", "double_review"}
    elif quality_gate in {
        "paper_v3_dual_ai_recovery",
        "paper_v3_sequential_multihop_v2",
    }:
        # Recovery rows deliberately remain ``draft`` so AI review is never
        # confused with human gold.  The exact reviewed manifest is bound and
        # validated by the experiment preflight before this gate is accepted.
        allowed_statuses = {"draft", "double_review"}
    else:
        allowed_statuses = {"double_review"}
    bad_statuses = sorted({example.review_status for example in examples} - allowed_statuses)
    if bad_statuses:
        raise ValueError(f"SFT dataset violates {quality_gate} review status: {bad_statuses}")
    for task_id, steps in task_steps.items():
        if sorted(steps) != list(range(len(steps))):
            raise ValueError(f"SFT task {task_id} has a missing or duplicate trajectory step")
    if expected_tasks is not None and len(task_steps) != expected_tasks:
        raise ValueError(
            f"SFT task count {len(task_steps)} does not match preflight task count {expected_tasks}"
        )
    records = tuple(
        {
            "prompt": list(example.messages[:2]),
            "completion": [example.messages[2]],
            "example_id": example.example_id,
            "task_id": example.task_id,
        }
        for example in examples
    )
    audit = SFTDatasetAudit(
        path=str(dataset_path),
        sha256=sha256_file(dataset_path),
        bytes=dataset_path.stat().st_size,
        examples=len(examples),
        tasks=len(task_steps),
        source_papers=len({paper for example in examples for paper in example.source_paper_ids}),
        splits=dict(sorted(Counter(example.split for example in examples).items())),
        review_statuses=dict(sorted(Counter(example.review_status for example in examples).items())),
        actions=dict(sorted(action_counts.items())),
        policy_skill_name=policy_skill_name,
        policy_skill_sha256=policy_skill_sha256,
    )
    return audit, records


def validate_sft_paper_disjointness(
    training_path: str | Path,
    validation_path: str | Path,
) -> None:
    """Reject teacher-forced validation that reuses a training source paper."""

    def papers(path: str | Path) -> set[str]:
        result: set[str] = set()
        for row in _load_rows(Path(path).resolve()):
            source_papers = row.get("source_paper_ids")
            if not isinstance(source_papers, list) or any(
                not isinstance(value, str) or not value for value in source_papers
            ):
                raise ValueError("SFT dataset requires valid source_paper_ids")
            result.update(source_papers)
        return result

    overlap = sorted(papers(training_path) & papers(validation_path))
    if overlap:
        raise ValueError(f"SFT train/validation source papers overlap: {overlap}")


def resolve_training_output(repo_root: str | Path, configured: str) -> Path:
    root = Path(repo_root).resolve()
    relative = Path(configured)
    if relative.is_absolute():
        raise ValueError("training.output_dir must be repository-relative")
    output = (root / relative).resolve()
    checkpoint_root = (root / "runs" / "checkpoints").resolve()
    if output == checkpoint_root or checkpoint_root not in output.parents:
        raise ValueError("training.output_dir must be inside runs/checkpoints/<experiment>")
    return output


def fingerprint_artifact_tree(path: str | Path) -> dict[str, dict[str, int | str]]:
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"artifact directory is missing: {root}")
    result: dict[str, dict[str, int | str]] = {}
    for file in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        result[file.relative_to(root).as_posix()] = {
            "sha256": digest,
            "bytes": file.stat().st_size,
        }
    if not result:
        raise ValueError("artifact directory contains no files")
    return result
