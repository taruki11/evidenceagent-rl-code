"""External evidence-sufficiency verifier interfaces.

This module is deliberately dependency-light: importing it must not import torch, transformers, or
any GPU library.  The local implementation loads those libraries lazily inside ``load()``; the
harness only sees the small ``SufficiencyVerifier`` protocol and plain dataclasses.

The verifier reads only *public* policy state: the question, currently selected spans, the number of
distinct selected documents, search-turn/budget state, and visible candidate titles.  It must never
be given hidden TaskSpec fields or a regime label.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, Sequence


class SufficiencyLabel(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    UNANSWERABLE = "UNANSWERABLE"
    INSUFFICIENT = "INSUFFICIENT"


LABEL_ORDER = (
    SufficiencyLabel.SUFFICIENT,
    SufficiencyLabel.NEED_MORE_EVIDENCE,
    SufficiencyLabel.UNANSWERABLE,
)
BINARY_LABEL_ORDER = ("INSUFFICIENT", "SUFFICIENT")
_FORBIDDEN_REQUEST_KEYS = frozenset({
    "gold", "gold_answer", "supporting_facts", "regime", "answerable", "task_type",
})


class VerifierError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifierRequest:
    question: str
    selected_spans: tuple[dict[str, str], ...] = ()
    distinct_doc_count: int = 0
    search_turns: int = 0
    remaining_budget: int = 0
    candidate_titles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.question).strip():
            raise ValueError("question is required")
        if self.distinct_doc_count < 0 or self.search_turns < 0 or self.remaining_budget < 0:
            raise ValueError("counts must be non-negative")
        if len(self.selected_spans) > 0 and len({
            str(span.get("title", "")) for span in self.selected_spans
        }) > max(1, self.distinct_doc_count):
            raise ValueError("distinct_doc_count cannot be lower than selected titles")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "question": self.question,
            "selected_spans": [dict(s) for s in self.selected_spans],
            "distinct_doc_count": self.distinct_doc_count,
            "search_turns": self.search_turns,
            "remaining_budget": self.remaining_budget,
            "candidate_titles": list(self.candidate_titles),
        }
        if _FORBIDDEN_REQUEST_KEYS & set(payload):
            raise ValueError("public verifier request contained a hidden/gold field")
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VerifierRequest":
        forbidden = _FORBIDDEN_REQUEST_KEYS & set(payload)
        if forbidden:
            raise ValueError(f"forbidden verifier request fields: {sorted(forbidden)}")
        return cls(
            question=str(payload["question"]),
            selected_spans=tuple(
                {"title": str(s.get("title", "")), "text": str(s.get("text", ""))}
                for s in payload.get("selected_spans", [])
            ),
            distinct_doc_count=int(payload.get("distinct_doc_count", 0)),
            search_turns=int(payload.get("search_turns", 0)),
            remaining_budget=int(payload.get("remaining_budget", 0)),
            candidate_titles=tuple(str(t) for t in payload.get("candidate_titles", [])),
        )


@dataclass(frozen=True)
class VerifierOutput:
    label: SufficiencyLabel
    probabilities: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.label, SufficiencyLabel):
            object.__setattr__(self, "label", SufficiencyLabel(self.label))
        for value in self.probabilities.values():
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError("probabilities must be in [0, 1]")


class SufficiencyVerifier(Protocol):
    def score(self, request: VerifierRequest) -> VerifierOutput: ...


def format_verifier_text(request: VerifierRequest, *, binary: bool = False) -> str:
    """Canonical, gold-free text form shared by training and local inference."""
    lines = [f"Question: {request.question.strip()}", "Selected evidence:"]
    if not request.selected_spans:
        lines.append("- (none)")
    else:
        for span in request.selected_spans:
            title = str(span.get("title", "")).strip()
            text = str(span.get("text", "")).strip().replace("\n", " ")
            lines.append(f"- [{title}] {text}")
    lines.append(f"Distinct selected documents: {request.distinct_doc_count}")
    if not binary and request.candidate_titles:
        lines.append("Visible candidate titles: " + "; ".join(request.candidate_titles[:8]))
    return "\n".join(lines)


@dataclass(frozen=True)
class VerifierCalibration:
    temperature: float = 1.0
    sufficient_threshold: float = 0.5
    unanswerable_threshold: float = 0.5
    label_order: tuple[str, ...] = tuple(label.value for label in LABEL_ORDER)

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        for name in ("sufficient_threshold", "unanswerable_threshold"):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")

    def apply_temperature(self, logits: Sequence[float]) -> list[float]:
        scaled = [float(x) / self.temperature for x in logits]
        maximum = max(scaled)
        exp = [math.exp(x - maximum) for x in scaled]
        total = sum(exp)
        return [x / total for x in exp]

    def decide(self, probabilities: dict[str, float], *, reason: str = "") -> VerifierOutput:
        p_unans = float(probabilities.get(SufficiencyLabel.UNANSWERABLE.value, 0.0))
        p_suf = float(probabilities.get(SufficiencyLabel.SUFFICIENT.value, 0.0))
        if p_unans >= self.unanswerable_threshold and p_unans >= p_suf:
            label = SufficiencyLabel.UNANSWERABLE
        elif p_suf >= self.sufficient_threshold:
            label = SufficiencyLabel.SUFFICIENT
        else:
            label = SufficiencyLabel.NEED_MORE_EVIDENCE
        return VerifierOutput(label, dict(probabilities), reason=reason)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "VerifierCalibration":
        return cls(
            temperature=float(payload.get("temperature", 1.0)),
            sufficient_threshold=float(payload.get("sufficient_threshold", 0.5)),
            unanswerable_threshold=float(payload.get("unanswerable_threshold", 0.5)),
            label_order=tuple(payload.get("label_order", [x.value for x in LABEL_ORDER])),
        )


@dataclass(frozen=True)
class BinaryVerifierCalibration:
    """Temperature + one threshold for the binary SUFFICIENT/INSUFFICIENT head."""
    temperature: float = 1.0
    sufficient_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if not (0.0 <= float(self.sufficient_threshold) <= 1.0):
            raise ValueError("sufficient_threshold must be in [0, 1]")

    def apply_temperature(self, logits: Sequence[float]) -> list[float]:
        scaled = [float(x) / self.temperature for x in logits]
        maximum = max(scaled)
        exp = [math.exp(x - maximum) for x in scaled]
        total = sum(exp)
        return [x / total for x in exp]

    def decide(self, probabilities: dict[str, float], *, reason: str = "") -> VerifierOutput:
        p_suf = float(probabilities.get("SUFFICIENT", 0.0))
        label = (SufficiencyLabel.SUFFICIENT if p_suf >= self.sufficient_threshold
                 else SufficiencyLabel.INSUFFICIENT)
        return VerifierOutput(label, dict(probabilities), reason=reason)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "BinaryVerifierCalibration":
        return cls(
            temperature=float(payload.get("temperature", 1.0)),
            sufficient_threshold=float(payload.get("sufficient_threshold", 0.5)),
        )


class ScriptedSufficiencyVerifier:
    """Deterministic test/dry-run verifier; no model, no GPU."""

    def __init__(self, *, default: SufficiencyLabel = SufficiencyLabel.NEED_MORE_EVIDENCE,
                 labels: Sequence[SufficiencyLabel] | None = None,
                 decision_fn: Callable[[VerifierRequest], SufficiencyLabel] | None = None,
                 probabilities: dict[str, float] | None = None):
        self.default = SufficiencyLabel(default)
        self.labels = [SufficiencyLabel(x) for x in (labels or [])]
        self.decision_fn = decision_fn
        self.probabilities = probabilities or {}
        self.calls: list[VerifierRequest] = []

    def score(self, request: VerifierRequest) -> VerifierOutput:
        self.calls.append(request)
        if self.decision_fn is not None:
            label = SufficiencyLabel(self.decision_fn(request))
        elif self.labels:
            label = self.labels.pop(0)
        else:
            label = self.default
        probs = dict(self.probabilities)
        if not probs:
            probs = {x.value: 0.0 for x in LABEL_ORDER}
            probs[label.value] = 1.0
        return VerifierOutput(label, probs, reason="scripted")


class HTTPSufficiencyVerifier:
    """Small stdlib HTTP client for a separately served verifier."""

    def __init__(self, endpoint: str, *, timeout_seconds: float = 5.0):
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must be http(s)")
        self.endpoint = endpoint
        self.timeout_seconds = float(timeout_seconds)

    def score(self, request: VerifierRequest) -> VerifierOutput:
        body = json.dumps(request.to_dict(), ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
            raise VerifierError(f"verifier request failed: {exc}") from exc
        label = SufficiencyLabel(str(payload["label"]))
        probabilities = {str(k): float(v) for k, v in (payload.get("probabilities") or {}).items()}
        return VerifierOutput(label, probabilities, reason=str(payload.get("reason", "")))


class LocalSufficiencyVerifier:
    """Lazy local sequence-classification verifier (BGE reranker / XLM-R style).

    Importing this class does not import torch.  ``load()`` is explicit and is only called by the
    evaluation/training process, never by the zero-dependency harness core.
    """

    def __init__(self, *, model_dir: str, checkpoint_path: str,
                 calibration_path: str, max_length: int = 1536,
                 device: str | None = None, autocast_bf16: bool = False,
                 mode: str = "tri"):
        if mode not in {"tri", "binary"}:
            raise ValueError("verifier mode must be 'tri' or 'binary'")
        self.mode = mode
        self.model_dir = model_dir
        self.checkpoint_path = checkpoint_path
        self.calibration_path = calibration_path
        self.max_length = int(max_length)
        self.device = device
        self.autocast_bf16 = bool(autocast_bf16)
        self.tokenizer = None
        self.model = None
        payload = json.loads(open(calibration_path, encoding="utf-8").read())
        self.calibration = (
            BinaryVerifierCalibration.from_json(payload) if self.mode == "binary"
            else VerifierCalibration.from_json(payload)
        )

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        num_labels = len(BINARY_LABEL_ORDER) if self.mode == "binary" else len(LABEL_ORDER)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, num_labels=num_labels, ignore_mismatched_sizes=True,
        )
        state = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        resolved = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(resolved)
        self.device = resolved

    def score(self, request: VerifierRequest) -> VerifierOutput:
        if self.model is None or self.tokenizer is None:
            raise VerifierError("call load() before score()")
        import torch
        text = format_verifier_text(request, binary=self.mode == "binary")
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            if self.autocast_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = self.model(**inputs).logits[0].detach().float().cpu().tolist()
            else:
                logits = self.model(**inputs).logits[0].detach().float().cpu().tolist()
        probs = self.calibration.apply_temperature(logits)
        labels = BINARY_LABEL_ORDER if self.mode == "binary" else tuple(x.value for x in LABEL_ORDER)
        probabilities = {label: float(probs[i]) for i, label in enumerate(labels)}
        return self.calibration.decide(probabilities, reason="local_verifier")



# ---------------------------------------------------------------------------
# LLM-as-a-Judge baseline (Phase 7)
# ---------------------------------------------------------------------------

_LLM_JUDGE_SYSTEM = (
    "You are a strict evidence-sufficiency judge. Given a question and the submitted "
    "supporting spans, decide whether those spans contain enough evidence to answer the "
    "question. Return exactly one uppercase label, SUFFICIENT or INSUFFICIENT, and no "
    "other text."
)


@dataclass(frozen=True)
class LLMJudgeCall:
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    label: str
    parse_ok: bool
    status: int


class LLMJudgeStats:
    """Thread-safe per-call latency/token accounting for the judge baseline."""

    def __init__(self) -> None:
        self._calls: list[LLMJudgeCall] = []
        self._lock = threading.Lock()

    def record(self, call: LLMJudgeCall) -> None:
        with self._lock:
            self._calls.append(call)

    @property
    def calls(self) -> tuple[LLMJudgeCall, ...]:
        with self._lock:
            return tuple(self._calls)

    def summary(self) -> dict[str, Any]:
        calls = self.calls
        latencies = [c.latency_ms for c in calls]
        ordered = sorted(latencies)

        def pct(p: float) -> float:
            if not ordered:
                return 0.0
            index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * p) - 1))
            return float(ordered[index])

        return {
            "calls": len(calls),
            "latency_ms_p50": pct(0.50),
            "latency_ms_p95": pct(0.95),
            "latency_ms_max": float(max(latencies)) if latencies else 0.0,
            "prompt_tokens": sum(c.prompt_tokens for c in calls),
            "completion_tokens": sum(c.completion_tokens for c in calls),
            "total_tokens": sum(c.prompt_tokens + c.completion_tokens for c in calls),
            "parse_failures": sum(not c.parse_ok for c in calls),
            "labels": {
                "SUFFICIENT": sum(c.label == "SUFFICIENT" for c in calls),
                "INSUFFICIENT": sum(c.label == "INSUFFICIENT" for c in calls),
            },
        }


class LLMJudgeSufficiencyVerifier:
    """Zero-shot LLM-as-a-Judge binary sufficiency baseline over vLLM's OpenAI API.

    The public request is reduced to exactly question + submitted spans + distinct-document
    count.  No gold field, regime, candidate title, search-turn or budget field is sent.
    Greedy decoding is used; malformed output fails closed to INSUFFICIENT.
    """

    kind = "llm_judge"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 60.0,
        max_tokens: int = 8,
        api_key: str | None = None,
        api_key_env: str | None = None,
        disable_thinking: bool = True,
    ) -> None:
        if not str(base_url).startswith(("http://", "https://")):
            raise ValueError("base_url must be http(s)")
        if not str(model).strip():
            raise ValueError("model is required")
        if float(timeout_seconds) <= 0:
            raise ValueError("timeout_seconds must be positive")
        if int(max_tokens) <= 0:
            raise ValueError("max_tokens must be positive")
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.timeout_seconds = float(timeout_seconds)
        self.max_tokens = int(max_tokens)
        self.api_key = api_key if api_key is not None else (
            os.environ.get(api_key_env) if api_key_env else None
        )
        self.disable_thinking = bool(disable_thinking)
        self.stats = LLMJudgeStats()

    def _endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def _payload(self, request: VerifierRequest) -> dict[str, Any]:
        # Deliberately only question, submitted spans, and distinct count.  Candidate titles,
        # search turns and remaining budget are not sent to the external judge.
        spans = "\n".join(
            f"- [{str(s.get('title', '')).strip()}] {str(s.get('text', '')).strip()}"
            for s in request.selected_spans
        ) or "- (none)"
        user = (
            f"Question: {request.question.strip()}\n"
            f"Submitted supporting spans:\n{spans}\n"
            f"Distinct supporting documents: {request.distinct_doc_count}\n"
            "Label:"
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _LLM_JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_tokens,
            "n": 1,
            "stream": False,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def _post(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self._endpoint(), data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise VerifierError(f"LLM judge request failed: {type(exc).__name__}: {exc}") from exc
        return status, body

    def score(self, request: VerifierRequest) -> VerifierOutput:
        started = time.perf_counter()
        status, body = self._post(self._payload(request))
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = body.get("usage") if isinstance(body, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        try:
            content = body["choices"][0]["message"]["content"]
            text = str(content).strip()
        except (KeyError, IndexError, TypeError):
            text = ""
        if text == SufficiencyLabel.SUFFICIENT.value:
            label = SufficiencyLabel.SUFFICIENT
            parse_ok = True
            reason = "llm_judge"
        elif text == SufficiencyLabel.INSUFFICIENT.value:
            label = SufficiencyLabel.INSUFFICIENT
            parse_ok = True
            reason = "llm_judge"
        else:
            # Fail closed: a malformed/unparseable judge verdict is never treated as sufficient.
            label = SufficiencyLabel.INSUFFICIENT
            parse_ok = False
            reason = "llm_judge_parse_failure"
        self.stats.record(LLMJudgeCall(
            latency_ms=latency_ms, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, label=label.value,
            parse_ok=parse_ok, status=status,
        ))
        raw = {
            "kind": self.kind,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "parse_ok": parse_ok,
            "status": status,
        }
        if not parse_ok:
            raw["raw_content"] = text[:200]
        return VerifierOutput(
            label,
            {SufficiencyLabel.SUFFICIENT.value: 1.0 if parse_ok and label is SufficiencyLabel.SUFFICIENT else 0.0,
             SufficiencyLabel.INSUFFICIENT.value: 1.0 if label is SufficiencyLabel.INSUFFICIENT else 0.0},
            reason=reason,
            raw=raw,
        )


def build_verifier_from_spec(spec: dict[str, Any]) -> SufficiencyVerifier | None:
    """Optional config factory used by the runner for verifier arms."""
    raw = spec.get("sufficiency_verifier")
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError("sufficiency_verifier must be an object")
    kind = str(raw.get("kind", "")).strip()
    if kind == "http":
        return HTTPSufficiencyVerifier(
            str(raw["endpoint"]), timeout_seconds=float(raw.get("timeout_seconds", 5.0)))
    if kind == "llm_judge":
        return LLMJudgeSufficiencyVerifier(
            base_url=str(raw["base_url"]),
            model=str(raw["model"]),
            timeout_seconds=float(raw.get("timeout_seconds", 60.0)),
            max_tokens=int(raw.get("max_tokens", 8)),
            api_key=raw.get("api_key"),
            api_key_env=raw.get("api_key_env"),
            disable_thinking=bool(raw.get("disable_thinking", True)),
        )
    if kind == "local":
        return LocalSufficiencyVerifier(
            model_dir=str(raw["model_dir"]),
            checkpoint_path=str(raw["checkpoint_path"]),
            calibration_path=str(raw["calibration_path"]),
            max_length=int(raw.get("max_length", 1536)),
            device=raw.get("device"),
            autocast_bf16=bool(raw.get("autocast_bf16", False)),
            mode=str(raw.get("mode", "tri")),
        )
    raise ValueError(f"unknown verifier kind: {kind!r}")
