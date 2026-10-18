"""Immutable rollout-batch manifests for Harness-authoritative policy updates."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .experiment import sha256_file
from .rl import compute_group_advantages
from .trainer_bridge import TrainerBatch, trainer_batch_from_json


@dataclass(frozen=True)
class RolloutBatchEntry:
    path: str
    sha256: str
    bytes: int
    task_id: str
    rollouts: int
    samples: int
    action_tokens: int


@dataclass(frozen=True)
class RolloutBatchAudit:
    files: int
    tasks: int
    rollouts: int
    samples: int
    action_tokens: int
    zero_variance_groups: int
    minimum_reward: float
    maximum_reward: float


@dataclass(frozen=True)
class RolloutLearningSignalAudit:
    groups: int
    rollouts: int
    strict_success_rate: float
    mixed_outcome_groups: int
    all_failure_groups: int
    all_success_groups: int
    zero_advantage_groups: int
    reward_order_violations: int
    positive_failed_rollouts: int
    active_task_ids: tuple[str, ...]
    recovery_task_ids: tuple[str, ...]
    monitoring_task_ids: tuple[str, ...]
    ready_for_rl: bool

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutPolicyIdentity:
    served_model_name: str
    adapter_manifest_sha256: str
    service_config_sha256: str
    policy_skill_name: str
    policy_skill_sha256: str


@dataclass(frozen=True)
class RolloutBatchManifest:
    version: int
    epistemic_status: str
    experiment_id: str
    model_id: str
    model_revision: str
    memory_mode: str
    reward_mode: str
    advantage_method: str
    policy_identity: RolloutPolicyIdentity
    entries: tuple[RolloutBatchEntry, ...]
    audit: RolloutBatchAudit

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "epistemic_status": self.epistemic_status,
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "memory_mode": self.memory_mode,
            "reward_mode": self.reward_mode,
            "advantage_method": self.advantage_method,
            "policy_identity": asdict(self.policy_identity),
            "entries": [asdict(entry) for entry in self.entries],
            "audit": asdict(self.audit),
        }


def relabel_rollout_advantages(
    raw: object,
    *,
    experiment_id: str,
    advantage_method: str,
) -> dict[str, Any]:
    """Reuse frozen trajectories while recomputing only group advantages."""

    if not isinstance(raw, dict):
        raise ValueError("rollout payload must be an object")
    rows = raw.get("rollouts")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("rollout payload requires at least two rollout rows")
    batch = trainer_batch_from_json(raw.get("trainer_batch"))
    if not experiment_id.strip():
        raise ValueError("target experiment_id cannot be empty")

    rollout_ids: list[str] = []
    rewards: list[float] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"rollout row {index} must be an object")
        rollout_id = row.get("rollout_id")
        reward = row.get("reward")
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ValueError(f"rollout row {index} requires rollout_id")
        if not isinstance(reward, (int, float)) or isinstance(reward, bool):
            raise ValueError(f"rollout row {index} requires numeric reward")
        rollout_ids.append(rollout_id)
        rewards.append(float(reward))
    if len(set(rollout_ids)) != len(rollout_ids):
        raise ValueError("rollout row IDs must be unique")

    advantages = compute_group_advantages(rewards, method=advantage_method)
    advantage_by_rollout = dict(zip(rollout_ids, advantages))
    sample_rollouts = {sample.rollout_id for sample in batch.samples}
    if sample_rollouts != set(rollout_ids):
        raise ValueError("trainer batch rollout IDs do not match rollout rows")
    reward_by_rollout = dict(zip(rollout_ids, rewards))
    for sample in batch.samples:
        if not math.isclose(
            sample.trajectory_reward,
            reward_by_rollout[sample.rollout_id],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{sample.sample_id}: trainer reward differs from rollout row")

    transformed = deepcopy(raw)
    for row in transformed["rollouts"]:
        row["advantage_method"] = advantage_method
        row["advantage"] = advantage_by_rollout[row["rollout_id"]]
    transformed_batch = transformed["trainer_batch"]
    transformed_batch["experiment_id"] = experiment_id
    transformed_batch["advantage_method"] = advantage_method
    for sample in transformed_batch["samples"]:
        sample["advantage"] = advantage_by_rollout[sample["rollout_id"]]
    trainer_batch_from_json(transformed_batch)
    return transformed


def _sha256_identity(value: object, field: str) -> str:
    digest = str(value or "").strip()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"rollout payload {field} must be lowercase SHA-256")
    return digest


def load_rollout_payload(
    path: str | Path,
) -> tuple[str, TrainerBatch, RolloutPolicyIdentity]:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"rollout payload must be an object: {source}")
    status = raw.get("epistemic_status")
    if status not in {"draft_rollout_smoke", "review_gated_rollout"}:
        raise ValueError(f"rollout payload has invalid epistemic status: {source}")
    if "trainer_batch" not in raw:
        raise ValueError(f"rollout payload lacks trainer_batch: {source}")
    skill = raw.get("skill_contract")
    if not isinstance(skill, dict):
        raise ValueError(f"rollout payload lacks skill_contract: {source}")
    served_model_name = str(raw.get("served_model_name", "")).strip()
    policy_skill_name = str(skill.get("name", "")).strip()
    if not served_model_name or not policy_skill_name:
        raise ValueError(f"rollout payload policy aliases are empty: {source}")
    identity = RolloutPolicyIdentity(
        served_model_name=served_model_name,
        adapter_manifest_sha256=_sha256_identity(
            raw.get("adapter_manifest_sha256"), "adapter_manifest_sha256"
        ),
        service_config_sha256=_sha256_identity(
            raw.get("service_config_sha256"), "service_config_sha256"
        ),
        policy_skill_name=policy_skill_name,
        policy_skill_sha256=_sha256_identity(skill.get("sha256"), "policy skill sha256"),
    )
    return str(status), trainer_batch_from_json(raw["trainer_batch"]), identity


def audit_rollout_batches(
    batches: Iterable[TrainerBatch],
    *,
    experiment_id: str,
    model_id: str,
    model_revision: str,
    memory_mode: str,
    reward_mode: str,
    advantage_method: str,
    group_size: int,
) -> RolloutBatchAudit:
    batch_list = tuple(batches)
    if not batch_list:
        raise ValueError("rollout manifest requires at least one trainer batch")
    seen_samples: set[str] = set()
    seen_rollouts: set[str] = set()
    tasks: set[str] = set()
    rewards: list[float] = []
    action_tokens = 0
    zero_variance = 0
    for batch in batch_list:
        expected = {
            "experiment_id": experiment_id,
            "model_id": model_id,
            "model_revision": model_revision,
            "memory_mode": memory_mode,
            "reward_mode": reward_mode,
            "advantage_method": advantage_method,
        }
        for field, value in expected.items():
            if getattr(batch, field) != value:
                raise ValueError(f"trainer batch {batch.task_id} has mismatched {field}")
        rollout_ids = {sample.rollout_id for sample in batch.samples}
        if len(rollout_ids) != group_size:
            raise ValueError(
                f"trainer batch {batch.task_id} has {len(rollout_ids)} rollouts, expected {group_size}"
            )
        overlap = seen_rollouts & rollout_ids
        if overlap:
            raise ValueError(f"duplicate rollout IDs across trainer batches: {sorted(overlap)}")
        sample_ids = {sample.sample_id for sample in batch.samples}
        if overlap_samples := seen_samples & sample_ids:
            raise ValueError(f"duplicate sample IDs across trainer batches: {sorted(overlap_samples)}")
        seen_rollouts.update(rollout_ids)
        seen_samples.update(sample_ids)
        tasks.add(batch.task_id)
        action_tokens += sum(len(sample.old_logprobs) for sample in batch.samples)
        rollout_rewards = {
            sample.rollout_id: sample.trajectory_reward for sample in batch.samples
        }
        rewards.extend(rollout_rewards.values())
        sample_advantages = [sample.advantage for sample in batch.samples]
        rollout_advantages = {
            sample.rollout_id: sample.advantage for sample in batch.samples
        }
        if all(abs(value) <= 1e-12 for value in sample_advantages):
            zero_variance += 1
        if advantage_method == "grpo":
            mean = sum(rollout_advantages.values()) / len(rollout_advantages)
            if abs(mean) > 1e-6:
                raise ValueError(f"GRPO advantages are not group-centered for {batch.task_id}")
        elif advantage_method not in {
            "state_anchored_grpo_v1",
            "state_anchored_grpo_v2",
            "state_anchored_turn_credit_v3",
            "verifier_progress_turn_credit_v4",
            "hybrid_outcome_verifier_progress_v5",
            "multihop_graph_progress_v6",
        }:
            if advantage_method != "rloo":
                raise ValueError(f"unsupported advantage method: {advantage_method}")
        if not all(math.isfinite(value) for value in (*rewards, *sample_advantages)):
            raise ValueError("rollout rewards and advantages must be finite")
    return RolloutBatchAudit(
        files=len(batch_list),
        tasks=len(tasks),
        rollouts=len(seen_rollouts),
        samples=len(seen_samples),
        action_tokens=action_tokens,
        zero_variance_groups=zero_variance,
        minimum_reward=min(rewards),
        maximum_reward=max(rewards),
    )


def audit_rollout_learning_signal(
    batches: Iterable[TrainerBatch],
) -> RolloutLearningSignalAudit:
    """Route rollout groups by outcome mix before spending GPU time on RL."""

    batch_list = tuple(batches)
    if not batch_list:
        raise ValueError("learning-signal audit requires at least one trainer batch")
    active: list[str] = []
    recovery: list[str] = []
    monitoring: list[str] = []
    total_rollouts = 0
    successful_rollouts = 0
    zero_advantage_groups = 0
    reward_order_violations = 0
    positive_failed_rollouts = 0

    for batch in batch_list:
        by_rollout: dict[str, tuple[bool, float]] = {}
        sample_advantages: list[float] = []
        for sample in batch.samples:
            observed = (sample.strict_success, sample.trajectory_reward)
            previous = by_rollout.setdefault(sample.rollout_id, observed)
            if previous != observed:
                raise ValueError(
                    f"{batch.task_id}: rollout outcomes differ across action samples: "
                    f"{sample.rollout_id}"
                )
            sample_advantages.append(sample.advantage)
        if batch.advantage_method not in {
            "state_anchored_grpo_v1",
            "state_anchored_grpo_v2",
            "state_anchored_turn_credit_v3",
            "verifier_progress_turn_credit_v4",
            "hybrid_outcome_verifier_progress_v5",
            "multihop_graph_progress_v6",
        }:
            for rollout_id in by_rollout:
                labels = {
                    sample.advantage
                    for sample in batch.samples
                    if sample.rollout_id == rollout_id
                }
                if len(labels) != 1:
                    raise ValueError(
                        f"{batch.task_id}: rollout labels differ across action samples: "
                        f"{rollout_id}"
                    )
        outcomes = tuple(value[0] for value in by_rollout.values())
        rewards = tuple(value[1] for value in by_rollout.values())
        total_rollouts += len(outcomes)
        successful_rollouts += sum(outcomes)
        positive_failed_rollouts += sum(
            not success and reward > 0.0
            for success, reward in zip(outcomes, rewards)
        )
        if all(abs(value) <= 1e-12 for value in sample_advantages):
            zero_advantage_groups += 1

        if any(outcomes) and not all(outcomes):
            active.append(batch.task_id)
            successful_rewards = [
                reward for success, reward in zip(outcomes, rewards) if success
            ]
            failed_rewards = [
                reward for success, reward in zip(outcomes, rewards) if not success
            ]
            if min(successful_rewards) <= max(failed_rewards):
                reward_order_violations += 1
        elif any(outcomes):
            monitoring.append(batch.task_id)
        else:
            recovery.append(batch.task_id)

    return RolloutLearningSignalAudit(
        groups=len(batch_list),
        rollouts=total_rollouts,
        strict_success_rate=successful_rollouts / total_rollouts,
        mixed_outcome_groups=len(active),
        all_failure_groups=len(recovery),
        all_success_groups=len(monitoring),
        zero_advantage_groups=zero_advantage_groups,
        reward_order_violations=reward_order_violations,
        positive_failed_rollouts=positive_failed_rollouts,
        active_task_ids=tuple(sorted(active)),
        recovery_task_ids=tuple(sorted(recovery)),
        monitoring_task_ids=tuple(sorted(monitoring)),
        ready_for_rl=bool(active) and reward_order_violations == 0,
    )


def build_rollout_batch_manifest(
    paths: Iterable[str | Path],
    *,
    repo_root: str | Path,
    experiment_id: str,
    model_id: str,
    model_revision: str,
    memory_mode: str,
    reward_mode: str,
    advantage_method: str,
    group_size: int,
) -> tuple[RolloutBatchManifest, tuple[TrainerBatch, ...]]:
    root = Path(repo_root).resolve()
    resolved = tuple(sorted({Path(path).resolve() for path in paths}, key=str))
    statuses: set[str] = set()
    policy_identities: set[RolloutPolicyIdentity] = set()
    batches: list[TrainerBatch] = []
    entries: list[RolloutBatchEntry] = []
    for path in resolved:
        if not path.is_file():
            raise ValueError(f"rollout batch file is missing: {path}")
        status, batch, policy_identity = load_rollout_payload(path)
        statuses.add(status)
        policy_identities.add(policy_identity)
        batches.append(batch)
        try:
            display = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("rollout batches must live inside the repository") from exc
        rollout_ids = {sample.rollout_id for sample in batch.samples}
        entries.append(
            RolloutBatchEntry(
                path=display,
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
                task_id=batch.task_id,
                rollouts=len(rollout_ids),
                samples=len(batch.samples),
                action_tokens=sum(len(sample.old_logprobs) for sample in batch.samples),
            )
        )
    if len(statuses) != 1:
        raise ValueError("draft and review-gated rollout payloads cannot be mixed")
    if len(policy_identities) != 1:
        raise ValueError("rollout payloads were sampled from different policy identities")
    audit = audit_rollout_batches(
        batches,
        experiment_id=experiment_id,
        model_id=model_id,
        model_revision=model_revision,
        memory_mode=memory_mode,
        reward_mode=reward_mode,
        advantage_method=advantage_method,
        group_size=group_size,
    )
    manifest = RolloutBatchManifest(
        version=1,
        epistemic_status=next(iter(statuses)),
        experiment_id=experiment_id,
        model_id=model_id,
        model_revision=model_revision,
        memory_mode=memory_mode,
        reward_mode=reward_mode,
        advantage_method=advantage_method,
        policy_identity=next(iter(policy_identities)),
        entries=tuple(entries),
        audit=audit,
    )
    return manifest, tuple(batches)


def load_rollout_batch_manifest(
    path: str | Path,
    *,
    repo_root: str | Path,
) -> tuple[RolloutBatchManifest, tuple[TrainerBatch, ...]]:
    manifest_path = Path(path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("rollout manifest must be a version 1 object")
    entries = raw.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("rollout manifest entries must be non-empty")
    root = Path(repo_root).resolve()
    paths: list[Path] = []
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"rollout manifest entry {index} is invalid")
        candidate = (root / entry["path"]).resolve()
        if root != candidate and root not in candidate.parents:
            raise ValueError("rollout manifest path escapes the repository")
        if not candidate.is_file() or sha256_file(candidate) != entry.get("sha256"):
            raise ValueError(f"rollout manifest hash mismatch: {entry['path']}")
        paths.append(candidate)
    rebuilt, batches = build_rollout_batch_manifest(
        paths,
        repo_root=root,
        experiment_id=str(raw.get("experiment_id", "")),
        model_id=str(raw.get("model_id", "")),
        model_revision=str(raw.get("model_revision", "")),
        memory_mode=str(raw.get("memory_mode", "")),
        reward_mode=str(raw.get("reward_mode", "")),
        advantage_method=str(raw.get("advantage_method", "")),
        group_size=int(entries[0].get("rollouts", 0)),
    )
    if rebuilt.to_json() != raw:
        raise ValueError("rollout manifest metadata does not match authoritative batches")
    return rebuilt, batches


def validate_adapter_manifest(
    path: str | Path,
    *,
    repo_root: str | Path,
    model_id: str,
    model_revision: str,
) -> Path:
    """Verify an SFT or RL adapter handoff and every file hash."""

    manifest_path = Path(path).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_statuses = {
        "measured_training_run_not_evaluation_result",
        "measured_rl_update_not_evaluation_result",
    }
    if not isinstance(raw, dict) or raw.get("epistemic_status") not in allowed_statuses:
        raise ValueError("adapter manifest has invalid epistemic status")
    plan = raw.get("plan")
    artifacts = raw.get("adapter_artifacts")
    if not isinstance(plan, dict) or not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("adapter manifest lacks plan or artifact hashes")
    model = plan.get("model")
    if not isinstance(model, dict) or model.get("id") != model_id or model.get("revision") != model_revision:
        raise ValueError("adapter base model identity does not match RL config")
    root = Path(repo_root).resolve()
    output = Path(str(plan.get("output_dir", ""))).resolve()
    checkpoint_root = (root / "runs" / "checkpoints").resolve()
    if checkpoint_root not in output.parents:
        raise ValueError("adapter output path is outside runs/checkpoints")
    adapter = (output / "final_adapter").resolve()
    if output not in adapter.parents or not adapter.is_dir():
        raise ValueError("adapter directory is missing")
    actual_files = {
        file.relative_to(adapter).as_posix(): file
        for file in adapter.rglob("*")
        if file.is_file()
    }
    if set(actual_files) != set(artifacts):
        raise ValueError("adapter artifact file set does not match manifest")
    for relative, file in actual_files.items():
        expected = artifacts[relative]
        if not isinstance(expected, dict) or sha256_file(file) != expected.get("sha256"):
            raise ValueError(f"adapter artifact hash mismatch: {relative}")
        if file.stat().st_size != expected.get("bytes"):
            raise ValueError(f"adapter artifact size mismatch: {relative}")
    return adapter


def compute_torch_policy_loss(
    rows: Iterable[tuple[str, Any, Any, Any, float]],
    *,
    clip_epsilon: float,
    kl_coefficient: float,
) -> tuple[Any, dict[str, float]]:
    """Differentiable rollout-equal objective matching the tensor-free contract."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for differentiable policy loss") from exc
    if not 0 <= clip_epsilon < 1 or kl_coefficient < 0:
        raise ValueError("invalid clipping or KL coefficient")
    rollout_tokens: dict[str, list[Any]] = {}
    policy_values: list[Any] = []
    kl_values: list[Any] = []
    clipped = 0
    token_count = 0
    for rollout_id, new, old, reference, advantage_value in rows:
        if new.ndim != 1 or old.shape != new.shape or reference.shape != new.shape or new.numel() == 0:
            raise ValueError("policy log-prob tensors must be aligned non-empty vectors")
        advantage = new.new_tensor(float(advantage_value))
        ratio = torch.exp(new - old)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        policy_loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage)
        log_ratio = reference - new
        kl = torch.exp(log_ratio) - log_ratio - 1.0
        total = policy_loss + kl_coefficient * kl
        rollout_tokens.setdefault(rollout_id, []).append(total)
        policy_values.append(policy_loss.detach())
        kl_values.append(kl.detach())
        clipped += int((ratio.detach() != clipped_ratio.detach()).sum().item())
        token_count += int(ratio.numel())
    if not rollout_tokens:
        raise ValueError("policy loss requires at least one rollout")
    rollout_losses = [torch.cat(tokens).mean() for tokens in rollout_tokens.values()]
    loss = torch.stack(rollout_losses).mean()
    metrics = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(torch.cat(policy_values).mean().cpu()),
        "kl": float(torch.cat(kl_values).mean().cpu()),
        "clip_fraction": clipped / token_count,
        "action_tokens": float(token_count),
        "rollouts": float(len(rollout_losses)),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"non-finite RL metrics: {metrics}")
    return loss, metrics
