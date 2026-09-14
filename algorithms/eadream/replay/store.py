"""Crash-safe completed and live episode storage."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
from typing import TypedDict
from uuid import uuid4

import numpy as np

from .episode import (
    EpisodeBuilder,
    REQUIRED_FIELDS,
    validate_episode,
    validate_replay_episode,
)


class EpisodeRecord(TypedDict):
    episode_id: str
    filename: str
    length: int
    ordinal: int
    sha256: str


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_STATE_FIELDS = {
    "format_version",
    "capacity",
    "next_ordinal",
    "completed",
    "live",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(directory: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, os.O_RDONLY | flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class EpisodeStore:
    def __init__(self, directory: str | Path, *, capacity: int = 1_000_000) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self._index: dict[str, EpisodeRecord] = {}
        self._next_ordinal = 0
        self._live: EpisodeBuilder | None = None
        self._rebuild_index()
        self._enforce_capacity()

    def append_completed(self, episode: Mapping[str, np.ndarray]) -> EpisodeRecord:
        validate_replay_episode(episode, completed=True)
        episode_id = str(episode["episode_id"][0])
        self._validate_episode_id(episode_id)
        if episode_id in self._index:
            raise ValueError(f"completed episode {episode_id!r} already exists")
        publishing_live = self._live is not None and self._live.episode_id == episode_id
        if publishing_live:
            if not self._live.is_finished:
                raise ValueError("matching live episode must be finished before publication")
            live_snapshot = self._live.snapshot()
            if any(
                not np.array_equal(live_snapshot[key], episode[key])
                for key in REQUIRED_FIELDS
            ):
                raise ValueError("completed payload does not match the live episode")
        elif self._live is not None:
            self._validate_live(self._live)
        length = len(episode["reward"])
        filename = f"{episode_id}-{length}.npz"
        payload_path = self.directory / filename
        marker_path = self.directory / f"{episode_id}.complete.json"
        if payload_path.exists() or marker_path.exists():
            raise FileExistsError(f"episode publication path already exists for {episode_id}")

        payload_temporary = self.directory / f".{episode_id}-{uuid4().hex}.npz.tmp"
        marker_temporary = self.directory / (
            f".{episode_id}-{uuid4().hex}.complete.json.tmp"
        )
        try:
            with payload_temporary.open("xb") as handle:
                np.savez_compressed(
                    handle, **{key: episode[key] for key in REQUIRED_FIELDS}
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(payload_temporary, payload_path)
            _sync_directory(self.directory)

            record: EpisodeRecord = {
                "episode_id": episode_id,
                "filename": filename,
                "length": length,
                "ordinal": self._next_ordinal,
                "sha256": _sha256(payload_path),
            }
            encoded = json.dumps(
                record, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            with marker_temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(marker_temporary, marker_path)
            _sync_directory(self.directory)
        except BaseException:
            payload_temporary.unlink(missing_ok=True)
            marker_temporary.unlink(missing_ok=True)
            # If the payload was published but its completion marker was not,
            # remove the orphan so a later checkpoint cannot discover it.
            if payload_path.exists() and not marker_path.exists():
                payload_path.unlink(missing_ok=True)
                _sync_directory(self.directory)
            raise

        self._index[episode_id] = record
        self._next_ordinal += 1
        if publishing_live:
            self._live = None
        self._enforce_capacity()
        return dict(record)

    def remove_completed(self, episode_ids: set[str]) -> None:
        """Rollback records published as part of a failed checkpoint transaction."""
        records = [self._index[episode_id] for episode_id in episode_ids if episode_id in self._index]
        for record in records:
            self._index.pop(record["episode_id"], None)
        self._delete_records(records)

    def set_live(self, episode: EpisodeBuilder | None) -> None:
        if episode is not None:
            if type(episode) is not EpisodeBuilder:
                raise TypeError("live episode must be an EpisodeBuilder or None")
            self._validate_live(episode)
            if episode.episode_id in self._index:
                raise ValueError("live episode_id already exists in completed replay")
        self._live = episode
        self._enforce_capacity()

    def sampleable_episodes(self) -> tuple[Mapping[str, np.ndarray], ...]:
        completed = tuple(self._load_indexed(record) for record in self._index.values())
        if self._live is None:
            return completed
        live = self._validate_live(self._live)
        return completed + (live,)

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "capacity": self.capacity,
            "next_ordinal": self._next_ordinal,
            "completed": [dict(record) for record in self._index.values()],
            "live": None if self._live is None else self._live.state_dict(),
        }

    def load_state_dict(
        self, state: Mapping[str, object], *, recover_orphans: bool = True
    ) -> None:
        """Restore state, optionally retaining valid disk-only records.

        Process recovery keeps records durably published after a snapshot;
        checkpoint restore passes ``recover_orphans=False`` for an exact
        historical replay population.
        """
        if (
            not isinstance(state, Mapping)
            or type(state.get("format_version")) is not int
            or state.get("format_version") != 1
        ):
            raise ValueError("unsupported episode store state")
        if set(state) != _STATE_FIELDS:
            raise ValueError("episode store checkpoint fields differ")
        if type(state.get("capacity")) is not int:
            raise TypeError("episode store checkpoint capacity must be an integer")
        if state.get("capacity") != self.capacity:
            raise ValueError("episode store checkpoint capacity does not match")
        raw_records = state.get("completed")
        if not isinstance(raw_records, list):
            raise TypeError("episode store completed state must be a list")

        checkpoint_records: list[EpisodeRecord] = []
        seen_ids: set[str] = set()
        seen_ordinals: set[int] = set()
        for raw_record in raw_records:
            record = self._coerce_record(raw_record)
            if record["episode_id"] in seen_ids or record["ordinal"] in seen_ordinals:
                raise ValueError("episode store checkpoint contains duplicate records")
            self._validate_record(record)
            checkpoint_records.append(record)
            seen_ids.add(record["episode_id"])
            seen_ordinals.add(record["ordinal"])
        if checkpoint_records != sorted(
            checkpoint_records, key=lambda item: item["ordinal"]
        ):
            raise ValueError("episode store checkpoint records are not in ordinal order")

        checkpoint_next = state.get("next_ordinal")
        if type(checkpoint_next) is not int or checkpoint_next < 0:
            raise ValueError("episode store next_ordinal must be non-negative")
        if checkpoint_records and checkpoint_next <= max(
            record["ordinal"] for record in checkpoint_records
        ):
            raise ValueError("episode store next_ordinal is not monotonic")

        disk_index, disk_next = self._scan_index()
        recovered: dict[str, EpisodeRecord] = {
            record["episode_id"]: record for record in checkpoint_records
        }
        ordinal_owners = {
            record["ordinal"]: record["episode_id"] for record in checkpoint_records
        }
        orphan_records: list[EpisodeRecord] = []
        for record in disk_index.values():
            existing = recovered.get(record["episode_id"])
            if existing is not None:
                if existing != record:
                    raise ValueError("checkpoint and disk episode records disagree")
                continue
            owner = ordinal_owners.get(record["ordinal"])
            if owner is not None:
                raise ValueError(
                    f"duplicate episode ordinal {record['ordinal']} for {owner} and "
                    f"{record['episode_id']}"
                )
            if recover_orphans:
                recovered[record["episode_id"]] = record
                ordinal_owners[record["ordinal"]] = record["episode_id"]
            else:
                orphan_records.append(record)
        recovered_records = sorted(
            recovered.values(), key=lambda item: item["ordinal"]
        )

        raw_live = state.get("live")
        live: EpisodeBuilder | None = None
        if raw_live is not None:
            if not isinstance(raw_live, Mapping):
                raise TypeError("episode store live state must be a mapping")
            for name in ("episode_id", "collector_id", "environment_id"):
                if type(raw_live.get(name)) is not str:
                    raise TypeError(f"live episode {name} must be a string")
            if type(raw_live.get("action_dim")) is not int:
                raise TypeError("live episode action_dim must be an integer")
            live = EpisodeBuilder(
                raw_live["episode_id"],
                raw_live["action_dim"],
                collector_id=raw_live["collector_id"],
                environment_id=raw_live["environment_id"],
            )
            live.load_state_dict(raw_live)
            if live.is_finished:
                raise ValueError("episode store live checkpoint is finished")
            self._validate_live(live)
            if live.episode_id in seen_ids:
                raise ValueError("live checkpoint episode is already completed")
            if live.episode_id in recovered:
                live = None

        recovered_next = (
            max((record["ordinal"] for record in recovered_records), default=-1) + 1
        )
        next_ordinal = max(checkpoint_next, disk_next, recovered_next) if recover_orphans else checkpoint_next
        candidate_index = {
            record["episode_id"]: record for record in recovered_records
        }
        candidate_index, evicted = self._capacity_candidates(candidate_index, live)

        self._index = candidate_index
        self._live = live
        self._next_ordinal = next_ordinal
        self._delete_records(evicted)
        if not recover_orphans:
            self._delete_records(orphan_records)

    def _rebuild_index(self) -> None:
        self._index, self._next_ordinal = self._scan_index()

    def _scan_index(self) -> tuple[dict[str, EpisodeRecord], int]:
        records: list[EpisodeRecord] = []
        for marker_path in self.directory.glob("*.complete.json"):
            try:
                raw = json.loads(marker_path.read_text(encoding="utf-8"))
                record = self._coerce_record(raw)
                if marker_path.name != f"{record['episode_id']}.complete.json":
                    continue
                self._validate_record(record)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            records.append(record)

        records.sort(key=lambda item: (item["ordinal"], item["episode_id"]))
        ordinal_owners: dict[int, str] = {}
        index: dict[str, EpisodeRecord] = {}
        for record in records:
            owner = ordinal_owners.get(record["ordinal"])
            if owner is not None:
                raise ValueError(
                    f"duplicate episode ordinal {record['ordinal']} for {owner} and "
                    f"{record['episode_id']}"
                )
            ordinal_owners[record["ordinal"]] = record["episode_id"]
            index[record["episode_id"]] = record
        next_ordinal = (
            max((record["ordinal"] for record in index.values()), default=-1) + 1
        )
        return index, next_ordinal

    def _coerce_record(self, raw: object) -> EpisodeRecord:
        if type(raw) is not dict or set(raw) != {
            "episode_id",
            "filename",
            "length",
            "ordinal",
            "sha256",
        }:
            raise ValueError("completed episode record fields differ")
        episode_id = raw["episode_id"]
        filename = raw["filename"]
        length = raw["length"]
        ordinal = raw["ordinal"]
        sha256 = raw["sha256"]
        if type(episode_id) is not str or type(filename) is not str:
            raise TypeError("completed episode record paths must be strings")
        if type(length) is not int or length < 1:
            raise ValueError("completed episode record length must be positive")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("completed episode record ordinal must be non-negative")
        if type(sha256) is not str or _HASH.fullmatch(sha256) is None:
            raise ValueError("completed episode record SHA256 is invalid")
        self._validate_episode_id(episode_id)
        if filename != f"{episode_id}-{length}.npz":
            raise ValueError("completed episode record filename does not match")
        return {
            "episode_id": episode_id,
            "filename": filename,
            "length": length,
            "ordinal": ordinal,
            "sha256": sha256,
        }

    def _validate_record(self, record: EpisodeRecord) -> dict[str, np.ndarray]:
        marker_path = self.directory / f"{record['episode_id']}.complete.json"
        try:
            marker = self._coerce_record(
                json.loads(marker_path.read_text(encoding="utf-8"))
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("completed episode marker is invalid") from error
        if marker != record:
            raise ValueError("completed episode marker does not match index")
        payload_path = self.directory / record["filename"]
        if not payload_path.is_file() or _sha256(payload_path) != record["sha256"]:
            raise ValueError("completed episode payload hash does not match")
        episode = self._load_payload(payload_path)
        validate_replay_episode(episode, completed=True)
        if len(episode["reward"]) != record["length"]:
            raise ValueError("completed episode payload length does not match")
        if str(episode["episode_id"][0]) != record["episode_id"]:
            raise ValueError("completed episode payload identity does not match")
        return episode

    def _load_indexed(self, record: EpisodeRecord) -> Mapping[str, np.ndarray]:
        episode = self._validate_record(record)
        for value in episode.values():
            value.setflags(write=False)
        return episode

    @staticmethod
    def _load_payload(payload_path: Path) -> dict[str, np.ndarray]:
        try:
            with np.load(payload_path, allow_pickle=False) as archive:
                if tuple(archive.files) != REQUIRED_FIELDS:
                    raise ValueError("completed episode payload fields or order differ")
                stored = {key: archive[key] for key in REQUIRED_FIELDS}
                validate_episode(stored)
                episode = {key: stored[key].copy() for key in REQUIRED_FIELDS}
        except (OSError, ValueError) as error:
            raise ValueError("completed episode payload is not a valid NPZ") from error
        return episode

    @staticmethod
    def _validate_episode_id(episode_id: str) -> None:
        if _SAFE_ID.fullmatch(episode_id) is None:
            raise ValueError("episode_id must be a filesystem-safe token")

    def _enforce_capacity(self) -> None:
        kept, evicted = self._capacity_candidates(self._index, self._live)
        self._index = kept
        self._delete_records(evicted)

    def _capacity_candidates(
        self,
        index: Mapping[str, EpisodeRecord],
        live: EpisodeBuilder | None,
    ) -> tuple[dict[str, EpisodeRecord], list[EpisodeRecord]]:
        kept = dict(index)
        live_length = 0
        if live is not None and live.episode_id not in kept:
            live_length = len(self._validate_live(live)["reward"])
        completed_length = sum(record["length"] for record in kept.values())
        evicted: list[EpisodeRecord] = []
        while kept and completed_length + live_length > self.capacity:
            episode_id, record = next(iter(kept.items()))
            del kept[episode_id]
            completed_length -= record["length"]
            evicted.append(record)
        return kept, evicted

    def _delete_records(self, records: list[EpisodeRecord]) -> None:
        for record in records:
            episode_id = record["episode_id"]
            marker_path = self.directory / f"{episode_id}.complete.json"
            payload_path = self.directory / record["filename"]
            marker_path.unlink(missing_ok=True)
            _sync_directory(self.directory)
            payload_path.unlink(missing_ok=True)
            _sync_directory(self.directory)

    @staticmethod
    def _validate_live(episode: EpisodeBuilder) -> dict[str, np.ndarray]:
        if episode.is_finished:
            raise ValueError("live episode must be started and unfinished")
        try:
            snapshot = episode.snapshot()
        except RuntimeError as error:
            raise ValueError("live episode must be started and unfinished") from error
        validate_replay_episode(snapshot, completed=False)
        return snapshot


__all__ = ["EpisodeRecord", "EpisodeStore"]
