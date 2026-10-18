"""Read-only ROCm host inventory and explicit training-readiness assessment."""

from __future__ import annotations

import platform
import os
import importlib.metadata
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ROCmEnvironmentSnapshot:
    operating_system: str
    platform: str
    python_version: str
    torch_version: str | None
    torch_hip_version: str | None
    accelerator_available: bool
    accelerator_count: int
    accelerator_name: str | None
    accelerator_memory_gib: float | None
    bf16_supported: bool | None
    system_memory_gib: float | None
    disk_free_gib: float
    rocm_smi_available: bool
    rocm_smi_returncode: int | None
    rocm_smi_output: str
    torch_error: str | None = None
    package_versions: dict[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class ROCmEnvironmentReport:
    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    snapshot: ROCmEnvironmentSnapshot

    def to_json(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "snapshot": asdict(self.snapshot),
        }


def _system_memory_gib() -> float | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return page_size * pages / 1024**3


def collect_rocm_snapshot(workspace: str | Path = ".") -> ROCmEnvironmentSnapshot:
    """Inspect versions and capacity without allocating tensors or launching work."""

    root = Path(workspace).resolve()
    disk_free_gib = shutil.disk_usage(root).free / 1024**3
    torch_version: str | None = None
    hip_version: str | None = None
    available = False
    count = 0
    name: str | None = None
    memory_gib: float | None = None
    bf16: bool | None = None
    torch_error: str | None = None
    try:
        import torch  # type: ignore

        torch_version = str(torch.__version__)
        hip_version = str(torch.version.hip) if torch.version.hip else None
        available = bool(torch.cuda.is_available())
        if available:
            count = int(torch.cuda.device_count())
            properties = torch.cuda.get_device_properties(0)
            name = str(properties.name)
            memory_gib = float(properties.total_memory) / 1024**3
            bf16_probe = getattr(torch.cuda, "is_bf16_supported", None)
            bf16 = bool(bf16_probe()) if callable(bf16_probe) else None
    except (ImportError, RuntimeError, OSError) as exc:
        torch_error = f"{type(exc).__name__}: {exc}"

    rocm_smi = shutil.which("rocm-smi")
    smi_returncode: int | None = None
    smi_output = ""
    if rocm_smi:
        try:
            result = subprocess.run(
                [
                    rocm_smi,
                    "--showproductname",
                    "--showdriverversion",
                    "--showmeminfo",
                    "vram",
                    "--json",
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            smi_returncode = result.returncode
            smi_output = (result.stdout or result.stderr).strip()[:12000]
        except (OSError, subprocess.SubprocessError) as exc:
            smi_returncode = -1
            smi_output = f"{type(exc).__name__}: {exc}"

    package_versions: dict[str, str | None] = {}
    for package in (
        "torch", "transformers", "datasets", "trl", "peft", "accelerate", "trackio", "vllm"
    ):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return ROCmEnvironmentSnapshot(
        operating_system=platform.system(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        torch_version=torch_version,
        torch_hip_version=hip_version,
        accelerator_available=available,
        accelerator_count=count,
        accelerator_name=name,
        accelerator_memory_gib=memory_gib,
        bf16_supported=bf16,
        system_memory_gib=_system_memory_gib(),
        disk_free_gib=disk_free_gib,
        rocm_smi_available=rocm_smi is not None,
        rocm_smi_returncode=smi_returncode,
        rocm_smi_output=smi_output,
        torch_error=torch_error,
        package_versions=package_versions,
    )


def assess_rocm_snapshot(
    snapshot: ROCmEnvironmentSnapshot,
    *,
    minimum_gpu_memory_gib: float = 180.0,
    minimum_system_memory_gib: float = 180.0,
    minimum_disk_free_gib: float = 80.0,
) -> ROCmEnvironmentReport:
    blockers: list[str] = []
    warnings: list[str] = []
    if snapshot.operating_system != "Linux":
        blockers.append("training host must be Linux for the reserved ROCm image")
    if snapshot.torch_error:
        blockers.append(f"PyTorch probe failed: {snapshot.torch_error}")
    if not snapshot.torch_version:
        blockers.append("PyTorch is unavailable")
    if not snapshot.torch_hip_version:
        blockers.append("PyTorch does not report a HIP runtime")
    if not snapshot.accelerator_available or snapshot.accelerator_count < 1:
        blockers.append("PyTorch cannot access an accelerator")
    if snapshot.accelerator_memory_gib is None:
        blockers.append("GPU memory capacity could not be measured")
    elif snapshot.accelerator_memory_gib < minimum_gpu_memory_gib:
        blockers.append(
            f"GPU memory {snapshot.accelerator_memory_gib:.1f} GiB is below "
            f"the required {minimum_gpu_memory_gib:.1f} GiB"
        )
    if snapshot.bf16_supported is False:
        blockers.append("accelerator does not report bfloat16 support")
    elif snapshot.bf16_supported is None:
        warnings.append("bfloat16 support could not be queried")
    if snapshot.system_memory_gib is None:
        warnings.append("system memory capacity could not be measured")
    elif snapshot.system_memory_gib < minimum_system_memory_gib:
        blockers.append(
            f"system memory {snapshot.system_memory_gib:.1f} GiB is below "
            f"the required {minimum_system_memory_gib:.1f} GiB"
        )
    if snapshot.disk_free_gib < minimum_disk_free_gib:
        blockers.append(
            f"free disk {snapshot.disk_free_gib:.1f} GiB is below "
            f"the required {minimum_disk_free_gib:.1f} GiB"
        )
    if not snapshot.rocm_smi_available:
        blockers.append("rocm-smi is unavailable")
    elif snapshot.rocm_smi_returncode != 0:
        blockers.append(f"rocm-smi failed with code {snapshot.rocm_smi_returncode}")
    return ROCmEnvironmentReport(not blockers, tuple(blockers), tuple(warnings), snapshot)


def probe_rocm_environment(
    workspace: str | Path = ".",
    *,
    minimum_gpu_memory_gib: float = 180.0,
    minimum_system_memory_gib: float = 180.0,
    minimum_disk_free_gib: float = 80.0,
) -> ROCmEnvironmentReport:
    """Probe the ROCm host and judge it against the given thresholds.

    The defaults are the project's formal-scale requirement and are unchanged for every existing
    caller.  A caller may pass lower values, but that is a POLICY decision, not a convenience: the
    evidence and the reason belong in that experiment's own decision sheet, because the whole point of
    the default is to stop a workload from being launched onto a host that cannot hold it.
    """

    return assess_rocm_snapshot(
        collect_rocm_snapshot(workspace),
        minimum_gpu_memory_gib=minimum_gpu_memory_gib,
        minimum_system_memory_gib=minimum_system_memory_gib,
        minimum_disk_free_gib=minimum_disk_free_gib,
    )
