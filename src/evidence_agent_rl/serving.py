"""Auditable local vLLM serving plans and capability probes.

This module never launches a server. It binds an immutable base checkpoint or
a hashed adapter checkpoint, request-facing name, tokenizer revision, and chat
template before any GPU-backed process is allowed to exist.
"""

from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from .experiment import PLACEHOLDER_MARKERS, sha256_file
from .react import SampledToken, sampled_content_tokens
from .rl_training import validate_adapter_manifest


@dataclass(frozen=True)
class ModelServerConfig:
    version: int
    service_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    checkpoint_kind: str
    adapter_manifest: str | None
    served_model_name: str
    host: str
    port: int
    dtype: str
    max_model_len: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    max_lora_rank: int | None
    chat_template: str | None
    language_model_only: bool
    reasoning_parser: str | None
    require_logprobs: bool
    enforce_eager: bool
    package_versions: dict[str, str]


@dataclass(frozen=True)
class ModelServerPlan:
    config: ModelServerConfig
    adapter_dir: str | None
    adapter_manifest_sha256: str | None
    argv: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "version": 1,
            "epistemic_status": (
                "artifact_validated_launch_plan_not_running_service"
                if self.config.checkpoint_kind == "adapter"
                else "immutable_base_identity_plan_not_running_or_package_verified"
            ),
            "config": asdict(self.config),
            "adapter_dir": self.adapter_dir,
            "adapter_manifest_sha256": self.adapter_manifest_sha256,
            "argv": list(self.argv),
        }


@dataclass(frozen=True)
class ServiceProbeReport:
    served_model_name: str
    advertised_models: tuple[str, ...]
    generation_checked: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    sampled_tokens: int = 0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _required_string(raw: dict[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"serving.{field} must be a non-empty string")
    return value.strip()


def model_server_config_from_dict(raw: object) -> ModelServerConfig:
    """Validate an already parsed serving config."""

    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("model server config must be a version 1 object")
    packages = raw.get("package_versions")
    required_packages = {"torch", "vllm", "transformers", "peft"}
    if not isinstance(packages, dict) or set(packages) != required_packages or any(
        not isinstance(value, str) or not value.strip() for value in packages.values()
    ):
        raise ValueError("package_versions must exactly pin torch, vllm, transformers, and peft")
    host = _required_string(raw, "host")
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("model server must bind to loopback only")
    revision = _required_string(raw, "model_revision")
    tokenizer_revision = _required_string(raw, "tokenizer_revision")
    if revision.casefold() in {"main", "latest"} or tokenizer_revision.casefold() in {
        "main", "latest"
    }:
        raise ValueError("model and tokenizer revisions must be immutable")
    if any(marker in revision or marker in tokenizer_revision for marker in PLACEHOLDER_MARKERS):
        raise ValueError("model and tokenizer revisions still contain placeholders")
    port = raw.get("port")
    max_len = raw.get("max_model_len")
    tensor_parallel = raw.get("tensor_parallel_size")
    rank = raw.get("max_lora_rank")
    utilization = raw.get("gpu_memory_utilization")
    if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535:
        raise ValueError("serving.port must be an integer in [1024, 65535]")
    if not isinstance(max_len, int) or isinstance(max_len, bool) or not 1024 <= max_len <= 8192:
        raise ValueError("serving.max_model_len must be an integer in [1024, 8192]")
    if tensor_parallel != 1:
        raise ValueError("the current single-GPU plan requires tensor_parallel_size=1")
    checkpoint_kind = _required_string(raw, "checkpoint_kind")
    if checkpoint_kind not in {"base", "adapter"}:
        raise ValueError("serving.checkpoint_kind must be base or adapter")
    if not isinstance(utilization, (int, float)) or isinstance(utilization, bool) or not 0.1 <= utilization <= 0.9:
        raise ValueError("gpu_memory_utilization must be in [0.1, 0.9]")
    if raw.get("dtype") != "bfloat16":
        raise ValueError("the reserved AMD serving plan requires bfloat16")
    if raw.get("require_logprobs") is not True:
        raise ValueError("EvidenceAgent-RL serving requires token logprobs")
    model_id = _required_string(raw, "model_id")
    served_model_name = _required_string(raw, "served_model_name")
    if served_model_name == model_id:
        raise ValueError("served_model_name must be a distinct experiment alias")
    adapter_manifest = raw.get("adapter_manifest")
    if checkpoint_kind == "adapter":
        if not isinstance(adapter_manifest, str) or not adapter_manifest.strip():
            raise ValueError("adapter serving requires adapter_manifest")
        if rank not in {1, 8, 16, 32, 64, 128, 256, 320, 512}:
            raise ValueError("max_lora_rank is unsupported by vLLM")
        adapter_manifest = adapter_manifest.strip()
    else:
        if adapter_manifest is not None or rank is not None:
            raise ValueError("base serving forbids adapter_manifest and max_lora_rank")
    chat_template = raw.get("chat_template")
    if chat_template is not None and (not isinstance(chat_template, str) or not chat_template.strip()):
        raise ValueError("chat_template must be null or a non-empty repository path")
    language_model_only = raw.get("language_model_only", False)
    if not isinstance(language_model_only, bool):
        raise ValueError("language_model_only must be boolean")
    enforce_eager = raw.get("enforce_eager", False)
    if not isinstance(enforce_eager, bool):
        raise ValueError("enforce_eager must be boolean")
    reasoning_parser = raw.get("reasoning_parser")
    if reasoning_parser is not None and reasoning_parser != "qwen3":
        raise ValueError("reasoning_parser must be null or qwen3")
    return ModelServerConfig(
        version=1,
        service_id=_required_string(raw, "service_id"),
        model_id=model_id,
        model_revision=revision,
        tokenizer_revision=tokenizer_revision,
        checkpoint_kind=checkpoint_kind,
        adapter_manifest=adapter_manifest,
        served_model_name=served_model_name,
        host=host,
        port=port,
        dtype="bfloat16",
        max_model_len=max_len,
        gpu_memory_utilization=float(utilization),
        tensor_parallel_size=1,
        max_lora_rank=rank,
        chat_template=chat_template.strip() if isinstance(chat_template, str) else None,
        language_model_only=language_model_only,
        reasoning_parser=reasoning_parser,
        require_logprobs=True,
        enforce_eager=enforce_eager,
        package_versions={str(key): str(value) for key, value in packages.items()},
    )


def load_model_server_config(path: str | Path) -> ModelServerConfig:
    return model_server_config_from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def build_model_server_plan(
    config: ModelServerConfig, *, repo_root: str | Path
) -> ModelServerPlan:
    root = Path(repo_root).resolve()
    argv = [
        "vllm", "serve", config.model_id,
        "--host", config.host,
        "--port", str(config.port),
        "--revision", config.model_revision,
        "--tokenizer-revision", config.tokenizer_revision,
        "--served-model-name", (
            config.model_id if config.checkpoint_kind == "adapter" else config.served_model_name
        ),
        "--dtype", config.dtype,
        "--max-model-len", str(config.max_model_len),
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--tensor-parallel-size", str(config.tensor_parallel_size),
        "--generation-config", "vllm",
    ]
    adapter: Path | None = None
    manifest: Path | None = None
    if config.checkpoint_kind == "adapter":
        assert config.adapter_manifest is not None and config.max_lora_rank is not None
        manifest = (root / config.adapter_manifest).resolve()
        if root != manifest and root not in manifest.parents:
            raise ValueError("adapter manifest escapes the repository")
        adapter = validate_adapter_manifest(
            manifest, repo_root=root, model_id=config.model_id,
            model_revision=config.model_revision,
        )
        try:
            adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
            actual_rank = int(adapter_config["r"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("adapter_config.json must record the trained LoRA rank") from exc
        if actual_rank != config.max_lora_rank:
            raise ValueError(
                f"max_lora_rank {config.max_lora_rank} does not match trained adapter rank {actual_rank}"
            )
        argv.extend((
            "--enable-lora", "--max-lora-rank", str(config.max_lora_rank),
            "--lora-modules", f"{config.served_model_name}={adapter}",
        ))
    if config.chat_template:
        template = (root / config.chat_template).resolve()
        if root != template and root not in template.parents:
            raise ValueError("chat template escapes the repository")
        if not template.is_file():
            raise ValueError("chat template file is missing")
        argv.extend(("--chat-template", str(template)))
    if config.language_model_only:
        argv.append("--language-model-only")
    if config.reasoning_parser:
        argv.extend(("--reasoning-parser", config.reasoning_parser))
    if config.enforce_eager:
        argv.append("--enforce-eager")
    return ModelServerPlan(
        config=config,
        adapter_dir=str(adapter) if adapter is not None else None,
        adapter_manifest_sha256=sha256_file(manifest) if manifest is not None else None,
        argv=tuple(argv),
    )


def verify_installed_serving_packages(config: ModelServerConfig) -> dict[str, str]:
    """Require the live serving environment to match every pinned package."""

    installed: dict[str, str] = {}
    mismatches: list[str] = []
    for package, expected in config.package_versions.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = "<missing>"
        installed[package] = actual
        if actual != expected:
            mismatches.append(f"{package}: expected {expected}, got {actual}")
    if mismatches:
        raise ValueError("serving package lock mismatch: " + "; ".join(mismatches))
    return installed


def _request_json(url: str, *, payload: dict[str, Any] | None, api_key: str | None, timeout: int) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    data = None
    method = "GET"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        method = "POST"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def probe_model_service(
    *,
    base_url: str,
    served_model_name: str,
    api_key: str | None = None,
    check_generation: bool = False,
    timeout_seconds: int = 30,
) -> ServiceProbeReport:
    """Check the advertised experiment identity and exact sampled bytes."""

    base = base_url.rstrip("/")
    models = _request_json(base + "/v1/models", payload=None, api_key=api_key, timeout=timeout_seconds)
    data = models.get("data")
    if not isinstance(data, list):
        raise ValueError("model service returned an invalid /v1/models payload")
    advertised = tuple(
        str(item["id"]) for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)
    )
    if served_model_name not in advertised:
        raise ValueError(f"required experiment service name is not advertised: {served_model_name}")
    if not check_generation:
        return ServiceProbeReport(served_model_name, advertised, False)
    result = _request_json(
        base + "/v1/chat/completions",
        payload={
            "model": served_model_name,
            "messages": [{"role": "user", "content": "Return {} exactly."}],
            "temperature": 0.0,
            "max_tokens": 8,
            "chat_template_kwargs": {"enable_thinking": False},
            "logprobs": True,
            "top_logprobs": 0,
        },
        api_key=api_key,
        timeout=timeout_seconds,
    )
    if result.get("model") != served_model_name:
        raise ValueError("generation response model does not match the required experiment alias")
    choices = result.get("choices")
    usage = result.get("usage")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(usage, dict):
        raise ValueError("generation probe lacks one choice or token usage")
    choice = choices[0]
    content = ((choice.get("message") or {}).get("content")) if isinstance(choice, dict) else None
    rows = ((choice.get("logprobs") or {}).get("content")) if isinstance(choice, dict) else None
    if not isinstance(content, str) or not isinstance(rows, list) or not rows:
        raise ValueError("generation probe lacks content or sampled logprobs")
    raw_tokens: list[SampledToken] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("bytes"), list):
            raise ValueError("every sampled token must expose exact UTF-8 bytes")
        token = SampledToken(
            str(row.get("token", "")), float(row.get("logprob", math.nan)),
            tuple(row["bytes"]),
        )
        raw_tokens.append(token)
    tokens = sampled_content_tokens(content, tuple(raw_tokens))
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        raise ValueError("generation probe token usage must be integers")
    if completion_tokens != len(raw_tokens):
        raise ValueError("completion token count does not match sampled logprobs")
    return ServiceProbeReport(
        served_model_name, advertised, True, prompt_tokens, completion_tokens, len(tokens)
    )
