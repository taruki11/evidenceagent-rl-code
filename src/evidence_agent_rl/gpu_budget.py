"""Auditable cumulative GPU-hour reservations for the fixed 100-hour quota."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Callable


class GPUReservationStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class GPUReservation:
    experiment_id: str
    config_sha256: str
    reserved_hours: float
    status: GPUReservationStatus
    created_at_utc: str
    updated_at_utc: str
    actual_hours: float | None = None

    @property
    def committed_hours(self) -> float:
        if self.status is GPUReservationStatus.CANCELLED:
            return 0.0
        if self.status is GPUReservationStatus.COMPLETED:
            return float(self.actual_hours or 0.0)
        return self.reserved_hours


@dataclass(frozen=True)
class GPUBudgetLedger:
    version: int
    total_hours: float
    reservations: tuple[GPUReservation, ...]

    @property
    def committed_hours(self) -> float:
        return sum(item.committed_hours for item in self.reservations)

    @property
    def remaining_hours(self) -> float:
        return self.total_hours - self.committed_hours

    @property
    def overrun_hours(self) -> float:
        return max(0.0, -self.remaining_hours)

    def to_json(self) -> dict[str, object]:
        return {
            "version": self.version,
            "total_hours": self.total_hours,
            "committed_hours": self.committed_hours,
            "remaining_hours": self.remaining_hours,
            "overrun_hours": self.overrun_hours,
            "reservations": [
                {**asdict(item), "status": item.status.value} for item in self.reservations
            ],
        }


def _positive_hours(value: object, field: str, *, allow_zero: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    hours = float(value)
    if hours < 0 or (hours == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field} must be {qualifier}")
    return hours


def _validate_config_sha256(value: str) -> str:
    digest = value.strip()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("config_sha256 must be lowercase SHA-256")
    return digest


def ledger_from_json(raw: object) -> GPUBudgetLedger:
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("GPU budget ledger must be a version 1 object")
    total = _positive_hours(raw.get("total_hours"), "total_hours")
    raw_reservations = raw.get("reservations")
    if not isinstance(raw_reservations, list):
        raise ValueError("reservations must be an array")
    reservations: list[GPUReservation] = []
    seen: set[str] = set()
    for row in raw_reservations:
        if not isinstance(row, dict):
            raise ValueError("each reservation must be an object")
        experiment_id = str(row.get("experiment_id", "")).strip()
        config_sha256 = str(row.get("config_sha256", "")).strip()
        if not experiment_id or experiment_id in seen:
            raise ValueError("reservation experiment IDs must be non-empty and unique")
        try:
            config_sha256 = _validate_config_sha256(config_sha256)
        except ValueError as exc:
            raise ValueError(f"{experiment_id}: {exc}") from exc
        seen.add(experiment_id)
        status = GPUReservationStatus(str(row.get("status", "")))
        reserved = _positive_hours(row.get("reserved_hours"), "reserved_hours")
        actual_raw = row.get("actual_hours")
        actual = (
            None
            if actual_raw is None
            else _positive_hours(actual_raw, "actual_hours", allow_zero=True)
        )
        if status is GPUReservationStatus.COMPLETED and actual is None:
            raise ValueError(f"{experiment_id}: completed reservation requires actual_hours")
        if status is not GPUReservationStatus.COMPLETED and actual is not None:
            raise ValueError(f"{experiment_id}: actual_hours is only valid when completed")
        created = str(row.get("created_at_utc", "")).strip()
        updated = str(row.get("updated_at_utc", "")).strip()
        if not created or not updated:
            raise ValueError(f"{experiment_id}: timestamps are required")
        reservations.append(
            GPUReservation(
                experiment_id,
                config_sha256,
                reserved,
                status,
                created,
                updated,
                actual,
            )
        )
    ledger = GPUBudgetLedger(1, total, tuple(reservations))
    return ledger


def load_gpu_budget_ledger(path: str | Path) -> GPUBudgetLedger:
    return ledger_from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def reserve_gpu_hours(
    ledger: GPUBudgetLedger,
    *,
    experiment_id: str,
    config_sha256: str,
    hours: float,
    timestamp_utc: str,
) -> GPUBudgetLedger:
    experiment_id = experiment_id.strip()
    timestamp_utc = timestamp_utc.strip()
    requested = _positive_hours(hours, "hours")
    config_sha256 = _validate_config_sha256(config_sha256)
    if not experiment_id or not timestamp_utc:
        raise ValueError("experiment_id and timestamp_utc are required")
    existing = next(
        (item for item in ledger.reservations if item.experiment_id == experiment_id), None
    )
    if existing:
        if (
            existing.status is GPUReservationStatus.PLANNED
            and existing.config_sha256 == config_sha256
            and existing.reserved_hours == requested
        ):
            return ledger
        raise ValueError(f"experiment already exists in GPU budget ledger: {experiment_id}")
    if requested > ledger.remaining_hours + 1e-9:
        raise ValueError(
            f"GPU budget exceeded: requested {requested:.2f}h, "
            f"remaining {ledger.remaining_hours:.2f}h"
        )
    reservation = GPUReservation(
        experiment_id,
        config_sha256,
        requested,
        GPUReservationStatus.PLANNED,
        timestamp_utc,
        timestamp_utc,
    )
    return replace(ledger, reservations=ledger.reservations + (reservation,))


def transition_gpu_reservation(
    ledger: GPUBudgetLedger,
    *,
    experiment_id: str,
    target: GPUReservationStatus | str,
    timestamp_utc: str,
    actual_hours: float | None = None,
) -> GPUBudgetLedger:
    selected = GPUReservationStatus(target)
    timestamp_utc = timestamp_utc.strip()
    if not timestamp_utc:
        raise ValueError("timestamp_utc is required")
    index = next(
        (i for i, item in enumerate(ledger.reservations) if item.experiment_id == experiment_id),
        None,
    )
    if index is None:
        raise ValueError(f"unknown GPU reservation: {experiment_id}")
    current = ledger.reservations[index]
    allowed = {
        GPUReservationStatus.PLANNED: {
            GPUReservationStatus.RUNNING,
            GPUReservationStatus.CANCELLED,
        },
        GPUReservationStatus.RUNNING: {GPUReservationStatus.COMPLETED},
        GPUReservationStatus.COMPLETED: set(),
        GPUReservationStatus.CANCELLED: set(),
    }
    if selected not in allowed[current.status]:
        raise ValueError(f"invalid GPU reservation transition: {current.status} -> {selected}")
    actual: float | None = None
    if selected is GPUReservationStatus.COMPLETED:
        if actual_hours is None:
            raise ValueError("completing a reservation requires actual_hours")
        actual = _positive_hours(actual_hours, "actual_hours", allow_zero=True)
    elif actual_hours is not None:
        raise ValueError("actual_hours is only valid for completion")
    updated = replace(
        current,
        status=selected,
        updated_at_utc=timestamp_utc,
        actual_hours=actual,
    )
    reservations = list(ledger.reservations)
    reservations[index] = updated
    return replace(ledger, reservations=tuple(reservations))


def reservation_blocker(
    ledger: GPUBudgetLedger,
    *,
    experiment_id: str,
    config_sha256: str,
    requested_hours: float,
) -> str | None:
    reservation = next(
        (item for item in ledger.reservations if item.experiment_id == experiment_id), None
    )
    if reservation is None:
        return f"GPU hours are not reserved for experiment {experiment_id}"
    if reservation.status not in {GPUReservationStatus.PLANNED, GPUReservationStatus.RUNNING}:
        return f"GPU reservation is not launchable: {reservation.status.value}"
    if reservation.config_sha256 != config_sha256:
        return "GPU reservation config hash does not match the experiment config"
    if abs(reservation.reserved_hours - float(requested_hours)) > 1e-9:
        return "GPU reservation hours do not match runtime.max_gpu_hours"
    return None


class GPUBudgetLedgerStore:
    """Single-writer atomic updates with an explicit adjacent lock file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def read(self) -> GPUBudgetLedger:
        return load_gpu_budget_ledger(self.path)

    def update(self, transform: Callable[[GPUBudgetLedger], GPUBudgetLedger]) -> GPUBudgetLedger:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"GPU budget ledger is locked: {self.lock_path}") from exc
        os.close(descriptor)
        temporary: str | None = None
        try:
            updated = transform(self.read())
            ledger_from_json(updated.to_json())
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
                dir=self.path.parent,
                prefix=self.path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as stream:
                json.dump(updated.to_json(), stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                temporary = stream.name
            os.replace(temporary, self.path)
            temporary = None
            return updated
        finally:
            if temporary and Path(temporary).exists():
                Path(temporary).unlink()
            self.lock_path.unlink(missing_ok=True)
