"""Small, explicit objectives for state-anchored action preferences."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .react import JSONActionCodec


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _token_ids(value: Any, *, label: str) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise ValueError(f"tokenizer returned invalid {label} token IDs")
    return value


def tokenize_action_preference(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    max_sequence_length: int,
) -> dict[str, Any]:
    """Tokenize both actions and mask every prompt token from the loss."""

    pair_id = str(record.get("preference_id", ""))
    prompt = record.get("prompt")
    chosen = record.get("chosen")
    rejected = record.get("rejected")
    if not pair_id or not isinstance(prompt, list):
        raise ValueError("preference record lacks an ID or prompt")
    if not isinstance(chosen, list) or len(chosen) != 1:
        raise ValueError(f"{pair_id}: chosen must contain exactly one message")
    if not isinstance(rejected, list) or len(rejected) != 1:
        raise ValueError(f"{pair_id}: rejected must contain exactly one message")
    if chosen[0].get("role") != "assistant" or rejected[0].get("role") != "assistant":
        raise ValueError(f"{pair_id}: preference completions must be assistant messages")
    expected_state_sha = hashlib.sha256(_canonical_json(prompt).encode("utf-8")).hexdigest()
    if record.get("state_sha256") != expected_state_sha:
        raise ValueError(f"{pair_id}: public-state SHA-256 mismatch")

    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
        label="prompt",
    )
    prepared: dict[str, Any] = {
        "preference_id": pair_id,
        "task_id": str(record["task_id"]),
        "prompt_tokens": len(prompt_ids),
    }
    for branch, messages in (("chosen", chosen), ("rejected", rejected)):
        action_text = messages[0].get("content")
        if not isinstance(action_text, str):
            raise ValueError(f"{pair_id}: {branch} completion is not text")
        JSONActionCodec.decode(action_text)
        full_ids = _token_ids(
            tokenizer.apply_chat_template(
                prompt + messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            ),
            label=branch,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"{pair_id}: {branch} completion changes prompt tokenization")
        completion_ids = full_ids[len(prompt_ids) :]
        if not completion_ids:
            raise ValueError(f"{pair_id}: {branch} completion is fully masked")
        if len(full_ids) > max_sequence_length:
            raise ValueError(
                f"{pair_id}: {branch} has {len(full_ids)} tokens, above "
                f"max_sequence_length={max_sequence_length}"
            )
        if "<think>" in tokenizer.decode(completion_ids, skip_special_tokens=False):
            raise ValueError(f"{pair_id}: {branch} completion contains thinking tokens")
        prepared[f"{branch}_input_ids"] = full_ids
        prepared[f"{branch}_attention_mask"] = [1] * len(full_ids)
        prepared[f"{branch}_labels"] = [-100] * len(prompt_ids) + completion_ids
        prepared[f"{branch}_tokens"] = len(full_ids)
        prepared[f"{branch}_completion_tokens"] = len(completion_ids)
    return prepared


def audit_and_tokenize_action_preferences(
    records: Iterable[dict[str, Any]],
    tokenizer: Any,
    *,
    max_sequence_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    record_list = list(records)
    prepared = [
        tokenize_action_preference(
            record,
            tokenizer,
            max_sequence_length=max_sequence_length,
        )
        for record in record_list
    ]
    if not prepared:
        raise ValueError("preference dataset cannot be empty")
    return prepared, {
        "version": 1,
        "pairs": len(prepared),
        "unique_tasks": len({row["task_id"] for row in prepared}),
        "unique_states": len({record["state_sha256"] for record in record_list}),
        "max_sequence_length": max_sequence_length,
        "maximum_chosen_tokens": max(row["chosen_tokens"] for row in prepared),
        "maximum_rejected_tokens": max(row["rejected_tokens"] for row in prepared),
        "minimum_chosen_completion_tokens": min(
            row["chosen_completion_tokens"] for row in prepared
        ),
        "minimum_rejected_completion_tokens": min(
            row["rejected_completion_tokens"] for row in prepared
        ),
        "completion_only_masking": True,
        "enable_thinking": False,
    }


def completion_mean_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return one mean completion log-probability per causal-LM sequence."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must be aligned [batch, sequence, vocabulary]")
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    mask = shifted_labels.ne(-100)
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("preference sequence contains no supervised completion tokens")
    token_losses = F.cross_entropy(
        shifted_logits.transpose(1, 2),
        shifted_labels,
        reduction="none",
        ignore_index=-100,
    )
    return -(token_losses * mask).sum(dim=1) / counts


def state_anchored_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    """Return DPO loss for chosen/rejected actions sharing one public state."""

    tensors = (
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    if not tensors[0].numel():
        raise ValueError("state-anchored preference batch cannot be empty")
    if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
        raise ValueError("chosen, rejected, and reference log-probabilities must align")
    if beta <= 0:
        raise ValueError("beta must be positive")

    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = reference_chosen_logps - reference_rejected_logps
    return -F.logsigmoid(beta * (policy_margin - reference_margin)).mean()
