"""Tensor-free bridge from replayed Harness rollouts to policy-gradient samples."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from collections.abc import Mapping
from typing import Any, Protocol

from .react import POLICY_SYSTEM_PROMPT
from .rl import AdvantageMethod, CollectedRolloutGroup, RewardMode
from .sft import validate_public_context
from .policy_skill import validate_policy_skill_binding


@dataclass(frozen=True)
class TrainerActionSample:
    sample_id: str
    rollout_id: str
    task_id: str
    step_index: int
    messages: tuple[dict[str, str], ...]
    action_text: str
    sampled_token_text: tuple[str, ...]
    sampled_token_bytes: tuple[tuple[int, ...], ...]
    old_logprobs: tuple[float, ...]
    advantage: float
    trajectory_reward: float
    strict_success: bool
    reward_breakdown: dict[str, float]
    prompt_token_ids: tuple[int, ...] = ()
    sampled_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class TrainerBatch:
    version: int
    experiment_id: str
    model_id: str
    model_revision: str
    task_id: str
    memory_mode: str
    reward_mode: str
    advantage_method: str
    samples: tuple[TrainerActionSample, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "task_id": self.task_id,
            "memory_mode": self.memory_mode,
            "reward_mode": self.reward_mode,
            "advantage_method": self.advantage_method,
            "samples": [asdict(sample) for sample in self.samples],
        }


class TrainerTokenizer(Protocol):
    def __len__(self) -> int: ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> object: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


@dataclass(frozen=True)
class TokenizedTrainerAction:
    sample_id: str
    rollout_id: str
    task_id: str
    step_index: int
    prompt_token_ids: tuple[int, ...]
    action_token_ids: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    advantage: float
    trajectory_reward: float

    @property
    def action_loss_mask(self) -> tuple[bool, ...]:
        return tuple(True for _ in self.action_token_ids)


@dataclass(frozen=True)
class RolloutPolicyLoss:
    rollout_id: str
    action_tokens: int
    policy_loss: float
    kl_penalty: float
    total_loss: float
    clip_fraction: float


@dataclass(frozen=True)
class PolicyLossReport:
    rollouts: int
    action_tokens: int
    policy_loss: float
    kl_penalty: float
    total_loss: float
    clip_fraction: float
    per_rollout: tuple[RolloutPolicyLoss, ...]


def _validate_identity(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} cannot be empty")
    if any(marker in normalized for marker in ("REPLACE_", "PLACEHOLDER", "MODEL_ID_REQUIRED")):
        raise ValueError(f"{field} cannot contain a placeholder")
    return normalized


def _coerce_token_ids(value: object, *, field: str) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, tuple):
        value = list(value)
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = list(value[0])
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"tokenizer returned invalid {field}")
    return value


def build_trainer_batch(
    group: CollectedRolloutGroup,
    *,
    experiment_id: str,
    model_id: str,
    model_revision: str,
    memory_mode: str,
    reward_mode: RewardMode | str,
    advantage_method: AdvantageMethod | str,
) -> TrainerBatch:
    experiment_id = _validate_identity(experiment_id, "experiment_id")
    model_id = _validate_identity(model_id, "model_id")
    model_revision = _validate_identity(model_revision, "model_revision")
    if model_revision.casefold() in {"main", "latest"}:
        raise ValueError("model_revision must be immutable")
    reward_mode = RewardMode(reward_mode).value
    advantage_method = AdvantageMethod(advantage_method).value
    if not group.collected or not group.scored:
        raise ValueError("trainer batch requires a collected and scored rollout group")
    scored = {item.rollout.rollout_id: item for item in group.scored}
    if set(scored) != {item.candidate.rollout_id for item in group.collected}:
        raise ValueError("collected and scored rollout IDs do not match")
    task_ids = {item.candidate.task_id for item in group.collected}
    if len(task_ids) != 1:
        raise ValueError("trainer batch requires one task_id")
    task_id = next(iter(task_ids))
    samples: list[TrainerActionSample] = []
    for collected in group.collected:
        outcome = scored[collected.candidate.rollout_id]
        score = outcome.rollout.score
        reward_breakdown = {**asdict(score), "total": score.total}
        for step_index, decision in enumerate(collected.decisions):
            sampled = decision.sampled_tokens
            if not sampled and decision.raw_output == "" and decision.completion_tokens == 0:
                # An empty completion has no sampled policy tokens to optimize. Keep the
                # rollout reward on its preceding decisions instead of fabricating a
                # logprob for a token the server never generated.
                continue
            samples.append(
                TrainerActionSample(
                    sample_id=f"{collected.candidate.rollout_id}:decision-{step_index:02d}",
                    rollout_id=collected.candidate.rollout_id,
                    task_id=task_id,
                    step_index=step_index,
                    messages=(
                        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
                        {"role": "user", "content": decision.policy_context_json},
                    ),
                    action_text=decision.raw_output,
                    sampled_token_text=tuple(item.token for item in sampled),
                    sampled_token_bytes=tuple(item.utf8_bytes for item in sampled),
                    old_logprobs=tuple(item.logprob for item in sampled),
                    advantage=outcome.advantage,
                    trajectory_reward=outcome.rollout.reward,
                    strict_success=outcome.rollout.success,
                    reward_breakdown=reward_breakdown,
                    prompt_token_ids=decision.prompt_token_ids,
                    sampled_token_ids=tuple(
                        item.token_id for item in sampled if item.token_id is not None
                    ),
                )
            )
    batch = TrainerBatch(
        version=1,
        experiment_id=experiment_id,
        model_id=model_id,
        model_revision=model_revision,
        task_id=task_id,
        memory_mode=_validate_identity(memory_mode, "memory_mode"),
        reward_mode=reward_mode,
        advantage_method=advantage_method,
        samples=tuple(samples),
    )
    validate_trainer_batch(batch)
    return batch


def validate_trainer_batch(batch: TrainerBatch) -> None:
    if batch.version != 1 or not batch.samples:
        raise ValueError("trainer batch must be non-empty version 1")
    if len({sample.sample_id for sample in batch.samples}) != len(batch.samples):
        raise ValueError("trainer sample IDs must be unique")
    steps: dict[str, list[int]] = {}
    for sample in batch.samples:
        if sample.task_id != batch.task_id:
            raise ValueError("trainer sample task_id does not match batch")
        steps.setdefault(sample.rollout_id, []).append(sample.step_index)
        if tuple(message.get("role") for message in sample.messages) != ("system", "user"):
            raise ValueError(f"{sample.sample_id}: messages must be system,user")
        if sample.messages[0]["content"] != POLICY_SYSTEM_PROMPT:
            raise ValueError(f"{sample.sample_id}: policy system prompt changed")
        try:
            context = json.loads(sample.messages[1]["content"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"{sample.sample_id}: invalid policy context JSON") from exc
        validate_public_context(context)
        validate_policy_skill_binding(context)
        if not sample.old_logprobs:
            raise ValueError(f"{sample.sample_id}: sampled token logprobs are required")
        count = len(sample.old_logprobs)
        if len(sample.sampled_token_text) != count or len(sample.sampled_token_bytes) != count:
            raise ValueError(f"{sample.sample_id}: sampled token fields are misaligned")
        if any(not math.isfinite(value) for value in sample.old_logprobs):
            raise ValueError(f"{sample.sample_id}: old logprobs must be finite")
        has_prompt_ids = bool(sample.prompt_token_ids)
        has_sampled_ids = bool(sample.sampled_token_ids)
        if has_prompt_ids != has_sampled_ids:
            raise ValueError(
                f"{sample.sample_id}: prompt and sampled token IDs must be supplied together"
            )
        if has_sampled_ids and len(sample.sampled_token_ids) != count:
            raise ValueError(f"{sample.sample_id}: sampled token IDs and logprobs are misaligned")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (*sample.prompt_token_ids, *sample.sampled_token_ids)
        ):
            raise ValueError(f"{sample.sample_id}: token IDs must be non-negative integers")
        if not math.isfinite(sample.advantage) or not math.isfinite(sample.trajectory_reward):
            raise ValueError(f"{sample.sample_id}: advantage and reward must be finite")
        expected_reward_keys = {
            "task", "evidence", "citation", "abstain", "format", "tool_cost",
            "unsupported", "memory_violation", "token_cost", "total",
        }
        if set(sample.reward_breakdown) != expected_reward_keys or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in sample.reward_breakdown.values()
        ):
            raise ValueError(f"{sample.sample_id}: invalid reward breakdown")
        if any(not token_bytes for token_bytes in sample.sampled_token_bytes):
            raise ValueError(f"{sample.sample_id}: exact sampled token bytes are required")
        try:
            reconstructed = b"".join(
                bytes(token_bytes) for token_bytes in sample.sampled_token_bytes
            ).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{sample.sample_id}: sampled bytes are not UTF-8") from exc
        if reconstructed != sample.action_text:
            raise ValueError(f"{sample.sample_id}: sampled bytes do not reconstruct action")
    for rollout_id, indices in steps.items():
        if indices != sorted(set(indices)) or any(index < 0 for index in indices):
            raise ValueError(f"{rollout_id}: decision step indices must be unique and ordered")


def materialize_tokenized_actions(
    batch: TrainerBatch,
    tokenizer: TrainerTokenizer,
) -> tuple[TokenizedTrainerAction, ...]:
    """Tokenize with the trainer checkpoint and reject server/tokenizer drift."""

    validate_trainer_batch(batch)
    materialized: list[TokenizedTrainerAction] = []
    for sample in batch.samples:
        prompt_ids_raw = tokenizer.apply_chat_template(
            list(sample.messages),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        try:
            rendered_prompt_ids = _coerce_token_ids(
                prompt_ids_raw, field="chat template input_ids"
            )
        except ValueError as exc:
            raise ValueError(f"{sample.sample_id}: {exc}") from exc
        if not rendered_prompt_ids:
            raise ValueError(f"{sample.sample_id}: tokenizer returned invalid prompt IDs")
        if sample.prompt_token_ids:
            prompt_ids = rendered_prompt_ids
            if tuple(prompt_ids) != sample.prompt_token_ids:
                raise ValueError(f"{sample.sample_id}: trainer prompt IDs differ from server")
            action_ids = list(sample.sampled_token_ids)
            try:
                tokenizer_size = len(tokenizer)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{sample.sample_id}: trainer tokenizer has no valid vocabulary size"
                ) from exc
            if not isinstance(tokenizer_size, int) or tokenizer_size <= 0:
                raise ValueError(
                    f"{sample.sample_id}: trainer tokenizer has no valid vocabulary size"
                )
            if any(token_id >= tokenizer_size for token_id in action_ids):
                raise ValueError(
                    f"{sample.sample_id}: server action ID exceeds trainer vocabulary"
                )
        else:
            prompt_text = tokenizer.apply_chat_template(
                list(sample.messages),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if not isinstance(prompt_text, str):
                raise ValueError(
                    f"{sample.sample_id}: tokenizer returned invalid chat rendering"
                )
            try:
                prompt_ids = _coerce_token_ids(
                    tokenizer.encode(prompt_text, add_special_tokens=False),
                    field="prompt IDs",
                )
                full_ids = _coerce_token_ids(
                    tokenizer.encode(
                        prompt_text + sample.action_text, add_special_tokens=False
                    ),
                    field="full prompt and action IDs",
                )
            except ValueError as exc:
                raise ValueError(f"{sample.sample_id}: {exc}") from exc
            if prompt_ids != rendered_prompt_ids:
                raise ValueError(f"{sample.sample_id}: tokenizer prompt rendering drift")
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(f"{sample.sample_id}: action changes the prompt token prefix")
            action_ids = full_ids[len(prompt_ids) :]
        if not action_ids:
            raise ValueError(f"{sample.sample_id}: tokenizer returned invalid action IDs")
        if len(action_ids) != len(sample.old_logprobs):
            raise ValueError(
                f"{sample.sample_id}: trainer tokenizer action length {len(action_ids)} "
                f"does not match server logprobs {len(sample.old_logprobs)}"
            )
        materialized.append(
            TokenizedTrainerAction(
                sample.sample_id,
                sample.rollout_id,
                sample.task_id,
                sample.step_index,
                tuple(prompt_ids),
                tuple(action_ids),
                sample.old_logprobs,
                sample.advantage,
                sample.trajectory_reward,
            )
        )
    return tuple(materialized)


def trainer_batch_from_json(raw: object) -> TrainerBatch:
    """Load the frozen JSON boundary without silently coercing malformed fields."""

    if not isinstance(raw, dict):
        raise ValueError("trainer_batch must be an object")
    batch_fields = {
        "version", "experiment_id", "model_id", "model_revision", "task_id",
        "memory_mode", "reward_mode", "advantage_method", "samples",
    }
    if set(raw) != batch_fields:
        raise ValueError("trainer_batch fields differ from the frozen schema")
    if not isinstance(raw["samples"], list):
        raise ValueError("trainer_batch.samples must be an array")
    legacy_sample_fields = {
        "sample_id", "rollout_id", "task_id", "step_index", "messages",
        "action_text", "sampled_token_text", "sampled_token_bytes", "old_logprobs",
        "advantage", "trajectory_reward", "strict_success", "reward_breakdown",
    }
    exact_id_fields = {"prompt_token_ids", "sampled_token_ids"}
    samples: list[TrainerActionSample] = []
    for index, row in enumerate(raw["samples"], 1):
        if not isinstance(row, dict) or set(row) not in (
            legacy_sample_fields,
            legacy_sample_fields | exact_id_fields,
        ):
            raise ValueError(f"trainer sample {index} fields differ from the frozen schema")
        for field in ("sample_id", "rollout_id", "task_id", "action_text"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"trainer sample {index} requires string {field}")
        if not isinstance(row["step_index"], int) or isinstance(row["step_index"], bool):
            raise ValueError(f"trainer sample {index} requires integer step_index")
        if not isinstance(row["strict_success"], bool):
            raise ValueError(f"trainer sample {index} requires boolean strict_success")
        messages_raw = row["messages"]
        if not isinstance(messages_raw, list) or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not all(isinstance(value, str) for value in message.values())
            for message in messages_raw
        ):
            raise ValueError(f"trainer sample {index} has invalid messages")
        token_text = row["sampled_token_text"]
        token_bytes = row["sampled_token_bytes"]
        old_logprobs = row["old_logprobs"]
        prompt_token_ids = row.get("prompt_token_ids", [])
        sampled_token_ids = row.get("sampled_token_ids", [])
        if not isinstance(token_text, list) or any(not isinstance(value, str) for value in token_text):
            raise ValueError(f"trainer sample {index} has invalid sampled_token_text")
        if not isinstance(token_bytes, list) or any(
            not isinstance(values, list)
            or any(not isinstance(value, int) or isinstance(value, bool) for value in values)
            for values in token_bytes
        ):
            raise ValueError(f"trainer sample {index} has invalid sampled_token_bytes")
        if not isinstance(old_logprobs, list) or any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in old_logprobs
        ):
            raise ValueError(f"trainer sample {index} has invalid old_logprobs")
        for field, values in (
            ("prompt_token_ids", prompt_token_ids),
            ("sampled_token_ids", sampled_token_ids),
        ):
            if not isinstance(values, list) or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in values
            ):
                raise ValueError(f"trainer sample {index} has invalid {field}")
        reward_breakdown = row["reward_breakdown"]
        if not isinstance(reward_breakdown, dict):
            raise ValueError(f"trainer sample {index} has invalid reward_breakdown")
        for field in ("advantage", "trajectory_reward"):
            if not isinstance(row[field], (int, float)) or isinstance(row[field], bool):
                raise ValueError(f"trainer sample {index} requires numeric {field}")
        samples.append(
            TrainerActionSample(
                sample_id=row["sample_id"],
                rollout_id=row["rollout_id"],
                task_id=row["task_id"],
                step_index=row["step_index"],
                messages=tuple(dict(message) for message in messages_raw),
                action_text=row["action_text"],
                sampled_token_text=tuple(token_text),
                sampled_token_bytes=tuple(tuple(values) for values in token_bytes),
                old_logprobs=tuple(float(value) for value in old_logprobs),
                advantage=float(row["advantage"]),
                trajectory_reward=float(row["trajectory_reward"]),
                strict_success=row["strict_success"],
                reward_breakdown={key: float(value) for key, value in reward_breakdown.items()},
                prompt_token_ids=tuple(prompt_token_ids),
                sampled_token_ids=tuple(sampled_token_ids),
            )
        )
    for field in (
        "experiment_id", "model_id", "model_revision", "task_id",
        "memory_mode", "reward_mode", "advantage_method",
    ):
        if not isinstance(raw[field], str) or not raw[field]:
            raise ValueError(f"trainer_batch requires string {field}")
    if not isinstance(raw["version"], int) or isinstance(raw["version"], bool):
        raise ValueError("trainer_batch.version must be an integer")
    batch = TrainerBatch(
        version=raw["version"],
        experiment_id=raw["experiment_id"],
        model_id=raw["model_id"],
        model_revision=raw["model_revision"],
        task_id=raw["task_id"],
        memory_mode=raw["memory_mode"],
        reward_mode=raw["reward_mode"],
        advantage_method=raw["advantage_method"],
        samples=tuple(samples),
    )
    validate_trainer_batch(batch)
    return batch


def compute_clipped_policy_loss(
    actions: tuple[TokenizedTrainerAction, ...],
    *,
    new_logprobs: dict[str, tuple[float, ...]],
    reference_logprobs: dict[str, tuple[float, ...]] | None,
    clip_epsilon: float = 0.2,
    kl_coefficient: float = 0.02,
) -> PolicyLossReport:
    """Reference GRPO/RLOO loss math; the backend only supplies model logprobs.

    Token objectives are averaged inside each rollout and rollout losses are
    then averaged, so a longer interactive trace does not receive more weight
    merely because it contains more decisions or action tokens.
    """

    if not actions:
        raise ValueError("policy loss requires tokenized actions")
    if not 0.0 <= clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must be in [0, 1)")
    if not math.isfinite(kl_coefficient) or kl_coefficient < 0.0:
        raise ValueError("kl_coefficient must be finite and non-negative")
    sample_ids = {action.sample_id for action in actions}
    if set(new_logprobs) != sample_ids:
        raise ValueError("new_logprobs keys must exactly match trainer sample IDs")
    if kl_coefficient > 0.0:
        if reference_logprobs is None or set(reference_logprobs) != sample_ids:
            raise ValueError(
                "reference_logprobs must exactly match sample IDs when KL is enabled"
            )

    token_rows: dict[str, list[tuple[float, float, float, bool]]] = {}
    for action in actions:
        new_values = tuple(float(value) for value in new_logprobs[action.sample_id])
        old_values = action.old_logprobs
        if len(new_values) != len(old_values):
            raise ValueError(f"{action.sample_id}: new and old logprob lengths differ")
        reference_values = (
            tuple(float(value) for value in reference_logprobs[action.sample_id])
            if reference_logprobs is not None
            else tuple(0.0 for _ in old_values)
        )
        if len(reference_values) != len(old_values):
            raise ValueError(f"{action.sample_id}: reference logprob length differs")
        if not all(
            math.isfinite(value)
            for value in (*new_values, *old_values, *reference_values)
        ):
            raise ValueError(f"{action.sample_id}: logprobs must be finite")
        rows = token_rows.setdefault(action.rollout_id, [])
        for new, old, reference in zip(new_values, old_values, reference_values):
            try:
                ratio = math.exp(new - old)
            except OverflowError as exc:
                raise ValueError(f"{action.sample_id}: policy ratio overflow") from exc
            clipped_ratio = min(1.0 + clip_epsilon, max(1.0 - clip_epsilon, ratio))
            unclipped = ratio * action.advantage
            clipped = clipped_ratio * action.advantage
            policy_loss = -min(unclipped, clipped)
            kl = 0.0
            if kl_coefficient > 0.0:
                log_ratio = reference - new
                try:
                    kl = math.exp(log_ratio) - log_ratio - 1.0
                except OverflowError as exc:
                    raise ValueError(f"{action.sample_id}: KL estimate overflow") from exc
            total = policy_loss + kl_coefficient * kl
            if not all(math.isfinite(value) for value in (policy_loss, kl, total)):
                raise ValueError(f"{action.sample_id}: non-finite policy loss")
            rows.append((policy_loss, kl, total, ratio != clipped_ratio))

    per_rollout: list[RolloutPolicyLoss] = []
    for rollout_id, rows in token_rows.items():
        count = len(rows)
        per_rollout.append(
            RolloutPolicyLoss(
                rollout_id=rollout_id,
                action_tokens=count,
                policy_loss=sum(row[0] for row in rows) / count,
                kl_penalty=sum(row[1] for row in rows) / count,
                total_loss=sum(row[2] for row in rows) / count,
                clip_fraction=sum(row[3] for row in rows) / count,
            )
        )
    count = len(per_rollout)
    return PolicyLossReport(
        rollouts=count,
        action_tokens=sum(item.action_tokens for item in per_rollout),
        policy_loss=sum(item.policy_loss for item in per_rollout) / count,
        kl_penalty=sum(item.kl_penalty for item in per_rollout) / count,
        total_loss=sum(item.total_loss for item in per_rollout) / count,
        clip_fraction=sum(item.clip_fraction for item in per_rollout) / count,
        per_rollout=tuple(per_rollout),
    )
