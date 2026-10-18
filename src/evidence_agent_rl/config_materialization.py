"""Materialize an immutable experiment config from a read-only host report."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from .experiment import (
    PLACEHOLDER_MARKERS,
    experiment_config_from_dict,
    load_experiment_config,
    sha256_file,
)
from .rl_training import validate_adapter_manifest


def find_placeholders(value: object, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, str) and any(marker in value for marker in PLACEHOLDER_MARKERS):
        found.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            found.extend(find_placeholders(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_placeholders(child, f"{path}[{index}]"))
    return found


def _numeric_version(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)+)", value)
    if match is None:
        raise ValueError(f"package version is not comparable: {value}")
    return tuple(int(part) for part in match.group(1).split("."))


def require_minimum_version(versions: dict[str, Any], package: str, minimum: str) -> None:
    value = versions.get(package)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ROCm environment lacks required package: {package}")
    actual = _numeric_version(value)
    required = _numeric_version(minimum)
    width = max(len(actual), len(required))
    if actual + (0,) * (width - len(actual)) < required + (0,) * (width - len(required)):
        raise ValueError(f"{package} {value} is below required {minimum}")


def load_ready_environment_versions(
    report_path: str | Path,
    *,
    required_packages: set[str],
    purpose: str,
    minimum_versions: dict[str, str] | None = None,
) -> dict[str, str]:
    """Load exact package locks from a successful read-only ROCm report."""

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("probe_mode") != "read_only_no_tensor_allocation":
        raise ValueError("environment report was not produced by the read-only ROCm probe")
    if report.get("ready") is not True or report.get("blockers") != []:
        raise ValueError("ROCm environment report is not ready")
    snapshot = report.get("snapshot")
    versions = snapshot.get("package_versions") if isinstance(snapshot, dict) else None
    if not isinstance(versions, dict):
        raise ValueError("environment report lacks package_versions")
    all_required = required_packages | set(minimum_versions or {})
    missing = sorted(
        package for package in all_required
        if not isinstance(versions.get(package), str) or not versions[package].strip()
    )
    if missing:
        raise ValueError(f"ROCm environment lacks required {purpose} packages: {missing}")
    for package, minimum in (minimum_versions or {}).items():
        require_minimum_version(versions, package, minimum)
    return {package: str(versions[package]) for package in sorted(required_packages)}


def materialize_sft_config(
    template_path: str | Path,
    environment_report_path: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    template_file = Path(template_path).resolve()
    report_file = Path(environment_report_path).resolve()
    template = load_experiment_config(template_file)
    if template["stage"] not in {"sft_smoke", "sft_train"}:
        raise ValueError("only an SFT config can be materialized by this command")
    required = set(template["runtime"]["training_package_versions"])
    versions = load_ready_environment_versions(
        report_file,
        required_packages=required,
        purpose="SFT",
        minimum_versions={"transformers": "4.51.0"},
    )
    result = deepcopy(template)
    result["runtime"]["training_package_versions"] = {
        package: versions[package] for package in sorted(required)
    }
    try:
        report_display = report_file.relative_to(root).as_posix()
        template_display = template_file.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("template and environment report must live inside the repository") from exc
    result["provenance"] = {
        "materialized_from_template": template_display,
        "template_sha256": sha256_file(template_file),
        "environment_report": report_display,
        "environment_report_sha256": sha256_file(report_file),
    }
    if unresolved := find_placeholders(result):
        raise ValueError(f"materialized config still contains placeholders: {unresolved}")
    return result


def materialize_rl_config(
    template_path: str | Path,
    environment_report_path: str | Path,
    adapter_manifest_path: str | Path,
    *,
    repo_root: str | Path,
    runtime_backend: str | None = None,
) -> dict[str, Any]:
    """Fill a frozen RL template without requiring its future rollout file yet."""

    root = Path(repo_root).resolve()
    template_file = Path(template_path).resolve()
    report_file = Path(environment_report_path).resolve()
    adapter_file = Path(adapter_manifest_path).resolve()
    for label, path in (
        ("RL template", template_file),
        ("environment report", report_file),
        ("adapter manifest", adapter_file),
    ):
        if root != path and root not in path.parents:
            raise ValueError(f"{label} must live inside the repository")
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
    template = load_experiment_config(template_file)
    if template["stage"] not in {"rl_smoke", "rl_train"}:
        raise ValueError("only an RL config can be materialized by this command")
    model = template["model"]
    validate_adapter_manifest(
        adapter_file,
        repo_root=root,
        model_id=str(model["id"]),
        model_revision=str(model["revision"]),
    )
    required = set(template["runtime"]["training_package_versions"])
    versions = load_ready_environment_versions(
        report_file,
        required_packages=required,
        purpose="RL",
        minimum_versions={"transformers": "4.51.0"},
    )
    result = deepcopy(template)
    if runtime_backend is not None:
        if runtime_backend not in {"rocm", "cuda"}:
            raise ValueError("runtime_backend must be rocm or cuda")
        result["runtime"]["backend"] = runtime_backend
    result["model"]["adapter_manifest"] = adapter_file.relative_to(root).as_posix()
    result["runtime"]["training_package_versions"] = versions
    result["provenance"] = {
        "materialized_from_template": template_file.relative_to(root).as_posix(),
        "template_sha256": sha256_file(template_file),
        "environment_report": report_file.relative_to(root).as_posix(),
        "environment_report_sha256": sha256_file(report_file),
        "adapter_manifest_sha256": sha256_file(adapter_file),
    }
    if unresolved := find_placeholders(result):
        raise ValueError(f"materialized RL config still contains placeholders: {unresolved}")
    experiment_config_from_dict(result)
    return result
