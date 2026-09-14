"""Versioned, crash-safe EADream training checkpoints.

Checkpoint files are trusted local experiment artifacts.  They contain Python,
NumPy, and Torch objects by design and must not be loaded from untrusted
sources.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any
from uuid import uuid4

import numpy as np
import torch


SCHEMA_VERSION = 1
_HASH_LENGTH = 64
_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "config_hash",
        "provenance_hash",
        "world_model",
        "actor",
        "value",
        "slow_value",
        "optimizers",
        "scalers",
        "reward_ema",
        "harmonizers",
        "schedules",
        "counters",
        "replay",
        "rng",
        "action_meanings",
        "observation_contract_hash",
        "agent_interactions",
        "driver",
        "sampler",
        "policy_version",
        "update_counts",
        "resume_boundary",
        "resume_trace",
    }
)
_EXACT_FIELDS = _REQUIRED_FIELDS | frozenset({"replay_root"})


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot be trusted or is not compatible."""


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    sha256: str
    agent_interactions: int
    schema_version: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(directory: Path) -> None:
    # Windows does not generally support directory handles.  A best-effort
    # sync still gives POSIX filesystems a durable publication boundary.
    flags = getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, os.O_RDONLY | flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def capture_rng_state() -> dict[str, Any]:
    """Capture every process RNG consumed by the learner and sampler."""
    state: dict[str, Any] = {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    return state


def validate_rng_state(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping):
        raise CheckpointCompatibilityError("RNG state must be a mapping")
    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise CheckpointCompatibilityError("RNG state fields differ")
    try:
        random.Random().setstate(state["python"])
    except (TypeError, ValueError) as error:
        raise CheckpointCompatibilityError("Python RNG state is invalid") from error
    try:
        probe = np.random.RandomState()
        probe.set_state(state["numpy"])
    except (TypeError, ValueError) as error:
        raise CheckpointCompatibilityError("NumPy RNG state is invalid") from error
    if not isinstance(state["torch_cpu"], torch.Tensor):
        raise CheckpointCompatibilityError("Torch CPU RNG state is invalid")
    if state["torch_cpu"].dtype != torch.uint8 or state["torch_cpu"].ndim != 1:
        raise CheckpointCompatibilityError("Torch CPU RNG state is invalid")
    cuda_state = state["torch_cuda"]
    if cuda_state is not None and (
        not isinstance(cuda_state, list)
        or any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.ndim != 1
            for value in cuda_state
        )
    ):
        raise CheckpointCompatibilityError("Torch CUDA RNG state is invalid")


def restore_rng_state(state: Mapping[str, Any]) -> None:
    validate_rng_state(state)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _atomic_replace(source: Path, destination: Path) -> None:
    for delay in (0.05, 0.1, 0.2, 0.4, 0.8, None):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


class CheckpointManager:
    """Own publication, validation, and discovery of one checkpoint stream."""

    def __init__(
        self,
        directory: str | Path,
        *,
        config_hash: str,
        provenance_hash: str,
        schema_version: int = SCHEMA_VERSION,
        filename: str = "checkpoint.pt",
    ) -> None:
        candidate = Path(directory)
        if candidate.suffix and not candidate.is_dir() and filename == "checkpoint.pt":
            filename = candidate.name
            candidate = candidate.parent
        self.directory = candidate
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config_hash = self._validate_hash("config hash", config_hash)
        self.provenance_hash = self._validate_hash("provenance hash", provenance_hash)
        if type(schema_version) is not int or schema_version < 1:
            raise ValueError("schema_version must be a positive integer")
        self.schema_version = schema_version
        if type(filename) is not str or not filename.strip() or Path(filename).name != filename:
            raise ValueError("checkpoint filename must be a plain file name")
        self.filename = filename

    @staticmethod
    def _validate_hash(name: str, value: object) -> str:
        if type(value) is not str or len(value) != _HASH_LENGTH:
            raise ValueError(f"{name} must be a 64-character hexadecimal hash")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"{name} must be a 64-character hexadecimal hash") from error
        return value

    @property
    def path(self) -> Path:
        return self.directory / self.filename

    @property
    def metadata_path(self) -> Path:
        return self.directory / "metadata.json"

    def save(self, snapshot: Mapping[str, Any]) -> CheckpointRecord:
        if not isinstance(snapshot, Mapping):
            raise TypeError("snapshot must be a mapping")
        payload = copy.deepcopy(dict(snapshot))
        payload.setdefault("schema_version", self.schema_version)
        payload.setdefault("config_hash", self.config_hash)
        payload.setdefault("provenance_hash", self.provenance_hash)
        self._validate_payload(payload)
        destination = self.path
        temporary = self.directory / f".{destination.name}.{uuid4().hex}.tmp"
        previous_bytes = destination.read_bytes() if destination.exists() else None
        previous_metadata = {
            target: target.read_bytes() if target.exists() else None
            for target in (self.metadata_path, self._sidecar_path(destination))
        }
        published = False
        try:
            with temporary.open("xb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            # Validate the actual serialized bytes before touching the current
            # published checkpoint.  This also exercises the explicit
            # weights_only=False load boundary used by restore.
            written = self._read_payload(temporary)
            self._validate_payload(written)
            _atomic_replace(temporary, destination)
            published = True
            _sync_directory(self.directory)
            published_payload = self._read_payload(destination)
            self._validate_payload(published_payload)
            digest = _sha256(destination)
            interactions = payload["agent_interactions"]
            record = CheckpointRecord(
                path=destination,
                sha256=digest,
                agent_interactions=interactions,
                schema_version=self.schema_version,
            )
            metadata = {
                "checkpoint_path": destination.name,
                "sha256": digest,
                "agent_interactions": interactions,
                "schema_version": self.schema_version,
                "config_hash": self.config_hash,
                "provenance_hash": self.provenance_hash,
            }
            self._write_metadata(metadata, self.metadata_path)
            self._write_metadata(metadata, self._sidecar_path(destination))
            return record
        except BaseException:
            temporary.unlink(missing_ok=True)
            if published:
                try:
                    if previous_bytes is None:
                        destination.unlink(missing_ok=True)
                    else:
                        destination.write_bytes(previous_bytes)
                    for target, value in previous_metadata.items():
                        if value is None:
                            target.unlink(missing_ok=True)
                        else:
                            target.write_bytes(value)
                except OSError:
                    # The original publication boundary is still preserved
                    # whenever the filesystem permits recovery; surface the
                    # original write failure either way.
                    pass
            raise

    def load(
        self, path: str | Path, restore: Callable[[Mapping[str, Any]], None]
    ) -> CheckpointRecord:
        if not callable(restore):
            raise TypeError("restore must be callable")
        checkpoint_path = Path(path)
        metadata = self._read_metadata_for(checkpoint_path)
        record = self._validate_metadata(checkpoint_path, metadata)
        try:
            actual_hash = _sha256(checkpoint_path)
        except OSError as error:
            raise CheckpointCompatibilityError("checkpoint file is missing") from error
        if actual_hash != record.sha256:
            raise CheckpointCompatibilityError("checkpoint hash does not match metadata")
        payload = self._read_payload(checkpoint_path)
        self._validate_payload(payload)
        if payload["agent_interactions"] != record.agent_interactions:
            raise CheckpointCompatibilityError("checkpoint interaction count differs from metadata")
        # All validation is complete before invoking user code.  A corrupt or
        # incompatible file therefore cannot partially mutate the live agent.
        restore(payload)
        return record

    def latest_valid(self) -> CheckpointRecord | None:
        candidates: list[Path] = []
        if self.metadata_path.is_file():
            try:
                metadata = self._read_json(self.metadata_path)
                candidates.append(self.directory / str(metadata.get("checkpoint_path", "")))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        if self.path.is_file() and self.path not in candidates:
            candidates.append(self.path)
        for sidecar in self.directory.glob("*.metadata.json"):
            candidate = sidecar.with_name(sidecar.name[: -len(".metadata.json")])
            if candidate not in candidates:
                candidates.append(candidate)
        valid: list[CheckpointRecord] = []
        for candidate in candidates:
            try:
                metadata = self._read_metadata_for(candidate)
                record = self._validate_metadata(candidate, metadata)
                if _sha256(candidate) != record.sha256:
                    continue
                payload = self._read_payload(candidate)
                self._validate_payload(payload)
                if payload["agent_interactions"] != record.agent_interactions:
                    continue
                valid.append(record)
            except (OSError, ValueError, TypeError, json.JSONDecodeError, RuntimeError):
                continue
        if not valid:
            return None
        return max(valid, key=lambda record: (record.agent_interactions, record.path.name))

    @staticmethod
    def _sidecar_path(path: Path) -> Path:
        return path.with_name(path.name + ".metadata.json")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if type(value) is not dict:
            raise ValueError("checkpoint metadata must be an object")
        return value

    def _read_metadata_for(self, path: Path) -> dict[str, Any]:
        if self.metadata_path.is_file():
            try:
                canonical = self._read_json(self.metadata_path)
                if canonical.get("checkpoint_path") == path.name:
                    return canonical
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        sidecar = self._sidecar_path(path)
        if sidecar.is_file():
            try:
                return self._read_json(sidecar)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                # A torn sidecar must not hide an otherwise valid canonical
                # metadata record.
                pass
        if self.metadata_path.is_file():
            return self._read_json(self.metadata_path)
        raise CheckpointCompatibilityError("checkpoint metadata is missing")

    def _validate_metadata(self, path: Path, metadata: Mapping[str, Any]) -> CheckpointRecord:
        required = {
            "checkpoint_path",
            "sha256",
            "agent_interactions",
            "schema_version",
            "config_hash",
            "provenance_hash",
        }
        if set(metadata) != required:
            raise CheckpointCompatibilityError("checkpoint metadata fields differ")
        stored_name = metadata["checkpoint_path"]
        if type(stored_name) is not str or Path(stored_name).name != path.name:
            raise CheckpointCompatibilityError("checkpoint metadata path differs")
        try:
            digest = self._validate_hash("checkpoint hash", metadata["sha256"])
        except ValueError as error:
            raise CheckpointCompatibilityError(str(error)) from error
        if type(metadata["schema_version"]) is not int or metadata["schema_version"] != self.schema_version:
            raise CheckpointCompatibilityError("checkpoint schema version is incompatible")
        if metadata["config_hash"] != self.config_hash:
            raise CheckpointCompatibilityError("checkpoint config hash is incompatible")
        if metadata["provenance_hash"] != self.provenance_hash:
            raise CheckpointCompatibilityError("checkpoint provenance hash is incompatible")
        interactions = metadata["agent_interactions"]
        if type(interactions) is not int or interactions < 0:
            raise CheckpointCompatibilityError("checkpoint interaction count is invalid")
        return CheckpointRecord(path=path, sha256=digest, agent_interactions=interactions, schema_version=self.schema_version)

    @staticmethod
    def _read_payload(path: Path) -> Mapping[str, Any]:
        try:
            with path.open("rb") as handle:
                payload = torch.load(handle, map_location="cpu", weights_only=False)
        except Exception as error:
            raise CheckpointCompatibilityError("checkpoint payload cannot be loaded") from error
        if not isinstance(payload, Mapping):
            raise CheckpointCompatibilityError("checkpoint payload must be a mapping")
        return payload

    def _validate_payload(self, payload: Mapping[str, Any]) -> None:
        missing = _EXACT_FIELDS - set(payload)
        extra = set(payload) - _EXACT_FIELDS
        if missing or extra:
            raise CheckpointCompatibilityError(
                f"checkpoint payload fields differ; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if type(payload["schema_version"]) is not int or payload["schema_version"] != self.schema_version:
            raise CheckpointCompatibilityError("checkpoint schema version is incompatible")
        if payload["config_hash"] != self.config_hash:
            raise CheckpointCompatibilityError("checkpoint config hash is incompatible")
        if payload["provenance_hash"] != self.provenance_hash:
            raise CheckpointCompatibilityError("checkpoint provenance hash is incompatible")
        interactions = payload["agent_interactions"]
        if type(interactions) is not int or interactions < 0:
            raise CheckpointCompatibilityError("checkpoint interaction count is invalid")
        if type(payload["policy_version"]) is not int or payload["policy_version"] < 0:
            raise CheckpointCompatibilityError("checkpoint policy version is invalid")
        observation_hash = payload["observation_contract_hash"]
        if type(observation_hash) is not str or len(observation_hash) != 64:
            raise CheckpointCompatibilityError("checkpoint observation contract hash is invalid")
        try:
            int(observation_hash, 16)
        except ValueError as error:
            raise CheckpointCompatibilityError("checkpoint observation contract hash is invalid") from error
        if not isinstance(payload["action_meanings"], (tuple, list)) or any(
            type(value) is not str or not value.strip() for value in payload["action_meanings"]
        ):
            raise CheckpointCompatibilityError("checkpoint action meanings are invalid")
        if type(payload["resume_boundary"]) is not bool or not isinstance(payload["resume_trace"], list) or any(type(value) is not str for value in payload["resume_trace"]):
            raise CheckpointCompatibilityError("checkpoint resume boundary is invalid")
        if payload["replay_root"] is not None and type(payload["replay_root"]) is not str:
            raise CheckpointCompatibilityError("checkpoint replay root is invalid")
        if not isinstance(payload["update_counts"], Mapping) or set(payload["update_counts"]) != {"world_model", "behavior"}:
            raise CheckpointCompatibilityError("checkpoint update counts are invalid")
        if any(type(value) is not int or value < 0 for value in payload["update_counts"].values()):
            raise CheckpointCompatibilityError("checkpoint update counts are invalid")
        counters = payload["counters"]
        if not isinstance(counters, Mapping) or set(counters) != {
            "agent_interactions", "ale_frames", "reset_noop_frames",
            "forced_start_frames", "evaluation_interactions", "gradient_updates",
        }:
            raise CheckpointCompatibilityError("checkpoint counters are invalid")
        if any(type(value) is not int or value < 0 for value in counters.values()):
            raise CheckpointCompatibilityError("checkpoint counters are invalid")
        if counters["agent_interactions"] != interactions:
            raise CheckpointCompatibilityError("checkpoint interaction counters disagree")
        for field in ("optimizers", "scalers"):
            if not isinstance(payload[field], Mapping) or set(payload[field]) != {"world_model", "actor", "value"}:
                raise CheckpointCompatibilityError(f"checkpoint {field} fields differ")
        if not isinstance(payload["harmonizers"], Mapping) or not payload["harmonizers"]:
            raise CheckpointCompatibilityError("checkpoint harmonizer fields differ")
        if not isinstance(payload["reward_ema"], Mapping) or set(payload["reward_ema"]) != {"alpha", "values"}:
            raise CheckpointCompatibilityError("checkpoint reward EMA fields differ")
        driver = payload["driver"]
        if not isinstance(driver, Mapping) or set(driver) != {
            "format_version", "counters", "replay", "schedules", "sampler",
            "episode_ordinal", "did_first_reset", "episode_start", "resume_boundary", "resume_trace",
        }:
            raise CheckpointCompatibilityError("checkpoint driver fields differ")
        if type(driver["format_version"]) is not int or driver["format_version"] != 1:
            raise CheckpointCompatibilityError("checkpoint driver version is invalid")
        if type(driver["episode_ordinal"]) is not int or driver["episode_ordinal"] < 0:
            raise CheckpointCompatibilityError("checkpoint episode ordinal is invalid")
        if type(driver["did_first_reset"]) is not bool or type(driver["episode_start"]) is not bool:
            raise CheckpointCompatibilityError("checkpoint driver flags are invalid")
        if driver["counters"] != counters:
            raise CheckpointCompatibilityError("checkpoint driver counters disagree")
        if not _state_equal(driver["replay"], payload["replay"]) or not _state_equal(driver["sampler"], payload["sampler"]):
            raise CheckpointCompatibilityError("checkpoint replay or sampler copies disagree")
        replay = payload["replay"]
        if not isinstance(replay, Mapping) or set(replay) not in (
            {"format_version", "completed", "live"},
            {"format_version", "capacity", "next_ordinal", "completed", "live"},
        ):
            raise CheckpointCompatibilityError("checkpoint replay fields differ")
        durable_replay = "capacity" in replay
        if (payload["replay_root"] is None) == durable_replay:
            raise CheckpointCompatibilityError("checkpoint replay root and format disagree")
        sampler = payload["sampler"]
        if sampler is not None and (not isinstance(sampler, Mapping) or set(sampler) != {"format_version", "rng_state"}):
            raise CheckpointCompatibilityError("checkpoint sampler fields differ")
        if sampler is not None:
            if type(sampler["format_version"]) is not int or sampler["format_version"] != 1:
                raise CheckpointCompatibilityError("checkpoint sampler version is invalid")
            try:
                probe = np.random.RandomState()
                probe.set_state(sampler["rng_state"])
            except (TypeError, ValueError) as error:
                raise CheckpointCompatibilityError("checkpoint sampler RNG is invalid") from error
        records = replay["completed"]
        if not isinstance(records, list):
            raise CheckpointCompatibilityError("checkpoint replay records are invalid")
        previous_ordinal = -1
        for record in records:
            if not isinstance(record, Mapping) or set(record) != {"episode_id", "filename", "length", "ordinal", "sha256"}:
                raise CheckpointCompatibilityError("checkpoint replay record fields differ")
            if type(record["episode_id"]) is not str or type(record["filename"]) is not str:
                raise CheckpointCompatibilityError("checkpoint replay record identity is invalid")
            if type(record["length"]) is not int or record["length"] < 1:
                raise CheckpointCompatibilityError("checkpoint replay record length is invalid")
            if type(record["ordinal"]) is not int or record["ordinal"] <= previous_ordinal:
                raise CheckpointCompatibilityError("checkpoint replay record ordinal is invalid")
            previous_ordinal = record["ordinal"]
            if record["filename"] != f"{record['episode_id']}-{record['length']}.npz":
                raise CheckpointCompatibilityError("checkpoint replay record filename is invalid")
            if type(record["sha256"]) is not str or len(record["sha256"]) != 64:
                raise CheckpointCompatibilityError("checkpoint replay record hash is invalid")
            try:
                int(record["sha256"], 16)
            except ValueError as error:
                raise CheckpointCompatibilityError("checkpoint replay record hash is invalid") from error
        schedules = payload["schedules"]
        if not isinstance(schedules, Mapping) or set(schedules) != {"pretrain", "train_every"}:
            raise CheckpointCompatibilityError("checkpoint schedules are invalid")
        pretrain = schedules["pretrain"]
        if not isinstance(pretrain, Mapping) or set(pretrain) != {"_once"} or type(pretrain["_once"]) is not bool:
            raise CheckpointCompatibilityError("checkpoint pretrain schedule is invalid")
        cadence = schedules["train_every"]
        if cadence is not None:
            if not isinstance(cadence, Mapping) or set(cadence) != {"_last"}:
                raise CheckpointCompatibilityError("checkpoint cadence schedule is invalid")
            last = cadence["_last"]
            if last is not None and (type(last) not in (int, float) or isinstance(last, bool) or not np.isfinite(last)):
                raise CheckpointCompatibilityError("checkpoint cadence schedule is invalid")
        if not _state_equal(driver["schedules"], schedules):
            raise CheckpointCompatibilityError("checkpoint schedule copies disagree")
        if driver["resume_boundary"] != payload["resume_boundary"] or driver["resume_trace"] != payload["resume_trace"]:
            raise CheckpointCompatibilityError("checkpoint resume copies disagree")
        validate_rng_state(payload["rng"])


    def _write_metadata(self, metadata: Mapping[str, Any], destination: Path) -> None:
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            encoded = json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8")
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            _atomic_replace(temporary, destination)
            _sync_directory(self.directory)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def _state_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_state_equal(a, b) for a, b in zip(left, right))
    return left == right


__all__ = [
    "CheckpointCompatibilityError",
    "CheckpointManager",
    "CheckpointRecord",
    "SCHEMA_VERSION",
    "capture_rng_state",
    "restore_rng_state",
]
