"""Strict JSON ReAct runner over the common EvidenceHarness."""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .harness import EvidenceHarness
from .models import Action, EpisodeState, TaskSpec
from .policy_skill import PolicySkillContract, coerce_policy_skill


POLICY_SYSTEM_PROMPT = (
    "You are the policy inside EvidenceAgent-RL. Follow the supplied skill contract. "
    "Return exactly one JSON action object and no markdown or hidden reasoning."
)


def build_policy_context(
    harness: EvidenceHarness, instructions: str | PolicySkillContract
) -> dict[str, Any]:
    """Canonical public context shared by inference and SFT export."""
    trace = [
        {
            "action": {"name": step.action.name, "arguments": step.action.arguments},
            "observation": {
                "ok": step.observation.ok,
                "kind": step.observation.kind,
                "data": step.observation.data,
                "error": step.observation.error,
            },
        }
        for step in harness.state.trace
    ]
    if harness.policy_context_version in {"v3", "v4", "v5", "v6"}:
        trace = [
            {
                "index": step.index,
                "action_name": step.action.name,
                "observation": {
                    "ok": step.observation.ok,
                    "kind": step.observation.kind,
                    "error": step.observation.error,
                },
            }
            for step in harness.state.trace
        ]
    contract = coerce_policy_skill(instructions)
    context = {
        "instructions": contract.content,
        "skill_contract": contract.identity(),
        "policy_state": harness.render_policy_state(),
        "skills": harness.skill_catalog(),
        "trace": trace,
        "output_contract": {
            "type": "object",
            "required": ["name", "arguments"],
            "additionalProperties": False,
        },
    }
    if harness.policy_context_version in {"v3", "v4", "v5", "v6"}:
        context["policy_context_version"] = harness.policy_context_version
    return context


def serialize_policy_context(context: dict[str, Any]) -> str:
    """Canonical prompt payload shared by inference, SFT, and trainer replay."""

    return json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class DecisionProvider(Protocol):
    def next_action(self, context: dict[str, Any]) -> str | "ModelActionOutput": ...


@dataclass(frozen=True)
class SampledToken:
    token: str
    logprob: float
    utf8_bytes: tuple[int, ...] = ()
    token_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.token, str) or not math.isfinite(self.logprob):
            raise ValueError("sampled token and finite logprob are required")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 255
            for value in self.utf8_bytes
        ):
            raise ValueError("sampled token bytes must be integers in [0, 255]")
        if self.token_id is not None and (
            not isinstance(self.token_id, int)
            or isinstance(self.token_id, bool)
            or self.token_id < 0
        ):
            raise ValueError("sampled token_id must be a non-negative integer")


_TERMINAL_CONTROL_TOKENS = frozenset({"<|im_end|>"})


def sampled_content_tokens(
    content: str, sampled_tokens: tuple[SampledToken, ...]
) -> tuple[SampledToken, ...]:
    """Return exact content tokens while excluding contentless control rows."""

    expected = content.encode("utf-8")
    reconstructed = b""
    content_tokens: list[SampledToken] = []
    for token in sampled_tokens:
        token_bytes = bytes(token.utf8_bytes)
        if not token_bytes:
            if token.token == "":
                # vLLM can expose an anonymous zero-byte control row between
                # ordinary content rows. It is safe to discard only because the
                # final exact reconstruction below still has to match content.
                continue
            if (
                reconstructed == expected
                and token.token in _TERMINAL_CONTROL_TOKENS
            ):
                continue
            inferred = token.token.encode("utf-8")
            candidate = reconstructed + inferred
            if (
                inferred
                and len(reconstructed) < len(expected)
                and expected.startswith(candidate)
            ):
                token_bytes = inferred
                token = SampledToken(
                    token.token,
                    token.logprob,
                    tuple(inferred),
                    token.token_id,
                )
            else:
                raise ValueError(
                    "sampled token bytes cannot be safely inferred from response content: "
                    f"token={token.token!r}, reconstructed={len(reconstructed)}, "
                    f"expected={len(expected)}"
                )
        if len(reconstructed) < len(expected):
            candidate = reconstructed + token_bytes
            if not expected.startswith(candidate):
                raise ValueError(
                    "sampled token bytes do not reconstruct response content"
                )
            reconstructed = candidate
            content_tokens.append(token)
            continue
        try:
            decoded = token_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("sampled token bytes are not valid UTF-8") from exc
        if token.token not in _TERMINAL_CONTROL_TOKENS or decoded != token.token:
            raise ValueError("sampled token bytes do not reconstruct response content")
    if reconstructed != expected:
        raise ValueError("sampled token bytes do not reconstruct response content")
    return tuple(content_tokens)


@dataclass(frozen=True)
class ModelActionOutput:
    """One model response plus auditable API token usage."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    sampled_tokens: tuple[SampledToken, ...] = ()
    prompt_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("model action content must be a string")
        counts = (self.prompt_tokens, self.completion_tokens)
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in counts
        ):
            raise ValueError("model token counts must be non-negative integers")
        if not isinstance(self.sampled_tokens, tuple) or any(
            not isinstance(item, SampledToken) for item in self.sampled_tokens
        ):
            raise TypeError("sampled_tokens must be a tuple of SampledToken")
        if not isinstance(self.prompt_token_ids, tuple) or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in self.prompt_token_ids
        ):
            raise TypeError("prompt_token_ids must be a tuple of non-negative integers")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ActionDecodeError(ValueError):
    pass


class DecisionProviderError(RuntimeError):
    """A model service failed before returning an action."""

    pass


class JSONActionCodec:
    @staticmethod
    def decode(raw_output: str) -> Action:
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise ActionDecodeError(f"model output is not JSON: {exc.msg}") from exc
        if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
            raise ActionDecodeError("action must contain exactly name and arguments")
        if not isinstance(payload["name"], str) or not isinstance(payload["arguments"], dict):
            raise ActionDecodeError("action name must be a string and arguments an object")
        return Action(payload["name"], payload["arguments"])


@dataclass(frozen=True)
class PolicyDecision:
    raw_output: str
    action: Action
    decode_error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    policy_context_json: str = ""
    sampled_tokens: tuple[SampledToken, ...] = ()
    prompt_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ReActRunResult:
    state: EpisodeState
    decisions: tuple[PolicyDecision, ...]

    @property
    def token_count(self) -> int:
        return sum(
            decision.prompt_tokens + decision.completion_tokens
            for decision in self.decisions
        )


class ReActRunner:
    def __init__(
        self,
        provider: DecisionProvider,
        *,
        max_decisions: int = 12,
        raise_provider_errors: bool = False,
    ) -> None:
        if max_decisions <= 0:
            raise ValueError("max_decisions must be positive")
        self.provider = provider
        self.max_decisions = max_decisions
        self.raise_provider_errors = raise_provider_errors

    def run(
        self,
        harness: EvidenceHarness,
        task: TaskSpec,
        *,
        episode_id: str,
        instructions: str | PolicySkillContract | Callable[[EvidenceHarness], str | PolicySkillContract],
    ) -> ReActRunResult:
        harness.reset(task, episode_id=episode_id)
        decisions: list[PolicyDecision] = []
        for _ in range(self.max_decisions):
            # `instructions` may be a fixed text/contract (the historical path, unchanged) or a
            # callable that selects the text for this decision from public state (candidate C's
            # two-stage instruction).  The callable is resolved BEFORE the context is built, so the
            # evidence state, the action budget and the scorer are untouched either way.
            step_instructions = instructions(harness) if callable(instructions) else instructions
            context = build_policy_context(harness, step_instructions)
            context_json = serialize_policy_context(context)
            try:
                generated = self.provider.next_action(context)
            except DecisionProviderError as exc:
                if self.raise_provider_errors:
                    raise
                error = str(exc)[:4000]
                action = Action("__provider_error__", {"error": error})
                decisions.append(
                    PolicyDecision(
                        "",
                        action,
                        decode_error=error,
                        policy_context_json=context_json,
                    )
                )
                harness.step(action)
                break
            if isinstance(generated, str):
                output = ModelActionOutput(generated)
            elif isinstance(generated, ModelActionOutput):
                output = generated
            else:
                raise TypeError("decision provider must return str or ModelActionOutput")
            raw_output = output.content
            try:
                action = JSONActionCodec.decode(raw_output)
                decision = PolicyDecision(
                    raw_output,
                    action,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    policy_context_json=context_json,
                    sampled_tokens=output.sampled_tokens,
                    prompt_token_ids=output.prompt_token_ids,
                )
            except ActionDecodeError as exc:
                action = Action("__invalid_format__", {"raw_output": raw_output[:4000]})
                decision = PolicyDecision(
                    raw_output,
                    action,
                    decode_error=str(exc),
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    policy_context_json=context_json,
                    sampled_tokens=output.sampled_tokens,
                    prompt_token_ids=output.prompt_token_ids,
                )
            decisions.append(decision)
            harness.step(action)
            if harness.state.done:
                break
        if not harness.state.done:
            harness.step(Action("__decision_limit__", {}))
        return ReActRunResult(harness.state, tuple(decisions))

    @staticmethod
    def _context(
        harness: EvidenceHarness, instructions: str | PolicySkillContract
    ) -> dict[str, Any]:
        return build_policy_context(harness, instructions)


class OpenAICompatibleDecisionProvider:
    """Minimal `/v1/chat/completions` client suitable for a later local vLLM server."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: int = 120,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = None,
        logprobs: bool = False,
        return_token_ids: bool = False,
        structured_action_output: bool = False,
    ) -> None:
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if return_token_ids and not logprobs:
            raise ValueError("return_token_ids requires logprobs")
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        self.logprobs = logprobs
        self.return_token_ids = return_token_ids
        self.structured_action_output = structured_action_output

    @classmethod
    def from_environment(
        cls,
        *,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = None,
        logprobs: bool = False,
        return_token_ids: bool = False,
        structured_action_output: bool = False,
        timeout_seconds: int | None = None,
    ) -> "OpenAICompatibleDecisionProvider":
        api_key = os.environ.get(api_key_env) if api_key_env else None
        resolved_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else int(os.environ.get("EVIDENCE_AGENT_HTTP_TIMEOUT", "300"))
        )
        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=resolved_timeout,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=logprobs,
            return_token_ids=return_token_ids,
            structured_action_output=structured_action_output,
        )

    @staticmethod
    def _action_response_format(context: dict[str, Any]) -> dict[str, Any]:
        skills = context.get("skills")
        if not isinstance(skills, (list, tuple)) or not skills:
            raise ValueError("structured action output requires a non-empty skill catalog")
        branches = []
        for skill in skills:
            if not isinstance(skill, dict):
                raise ValueError("structured action output requires object skill entries")
            name = skill.get("name")
            parameters = skill.get("parameters")
            if not isinstance(name, str) or not name or not isinstance(parameters, dict):
                raise ValueError("structured action output requires skill names and schemas")
            branches.append(
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "const": name},
                        "arguments": parameters,
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                }
            )
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "evidence_agent_action",
                "schema": {"oneOf": branches},
            },
        }

    def next_action(self, context: dict[str, Any]) -> ModelActionOutput:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": POLICY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": serialize_policy_context(context),
                },
            ],
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 0
        if self.return_token_ids:
            payload["return_token_ids"] = True
        if self.structured_action_output:
            payload["response_format"] = self._action_response_format(context)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:4000]
            raise DecisionProviderError(
                f"model service HTTP {exc.code}: {body or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise DecisionProviderError(f"model service request failed: {exc.reason}") from exc
        response_model = result.get("model")
        if response_model is not None and response_model != self.model:
            raise ValueError("model service response identity does not match requested model")
        usage = result.get("usage") or {}
        choice = result["choices"][0]
        raw_sampled_tokens = ((choice.get("logprobs") or {}).get("content") or [])
        raw_token_ids = choice.get("token_ids") if self.return_token_ids else None
        prompt_token_ids = result.get("prompt_token_ids") if self.return_token_ids else None
        if self.return_token_ids:
            for name, value in (
                ("prompt_token_ids", prompt_token_ids),
                ("choice.token_ids", raw_token_ids),
            ):
                if not isinstance(value, list) or any(
                    not isinstance(item, int) or isinstance(item, bool) or item < 0
                    for item in value
                ):
                    raise ValueError(f"model service omitted valid {name}")
            if len(raw_token_ids) != len(raw_sampled_tokens):
                raise ValueError("model service token_ids and logprobs are misaligned")
        sampled_tokens = tuple(
            SampledToken(
                token=str(item["token"]),
                logprob=float(item["logprob"]),
                utf8_bytes=tuple(int(value) for value in (item.get("bytes") or ())),
                token_id=(raw_token_ids[index] if raw_token_ids is not None else None),
            )
            for index, item in enumerate(raw_sampled_tokens)
        )
        if sampled_tokens:
            sampled_tokens = sampled_content_tokens(
                str(choice["message"]["content"]), sampled_tokens
            )
        return ModelActionOutput(
            str(choice["message"]["content"]),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            sampled_tokens=sampled_tokens,
            prompt_token_ids=tuple(prompt_token_ids or ()),
        )
