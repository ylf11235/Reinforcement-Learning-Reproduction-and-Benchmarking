"""Source-compatible length-weighted EADream sequence sampling."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy

import numpy as np

from .episode import REQUIRED_FIELDS, validate_episode


EpisodeSource = Sequence[Mapping[str, np.ndarray]] | Callable[
    [], Sequence[Mapping[str, np.ndarray]]
]


class SequenceSampler:
    def __init__(self, episodes: EpisodeSource, *, seed: int) -> None:
        if not callable(episodes) and not isinstance(episodes, Sequence):
            raise TypeError("episodes must be a sequence or zero-argument provider")
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        self._episodes = episodes
        self.rng = np.random.RandomState(seed)

    def sample(self, batch_size: int, length: int) -> dict[str, np.ndarray]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(length) is not int or length < 1:
            raise ValueError("length must be a positive integer")
        episodes = self._validated_episodes()
        weights = np.asarray(
            [len(episode["reward"]) for episode in episodes], dtype=np.float64
        )
        probabilities = weights / weights.sum()

        sequences = [
            self._sample_sequence(episodes, probabilities, length)
            for _ in range(batch_size)
        ]
        return {
            key: np.ascontiguousarray(
                np.stack([sequence[key] for sequence in sequences], axis=0)
            )
            for key in REQUIRED_FIELDS
        }

    def state_dict(self) -> dict[str, object]:
        return {"format_version": 1, "rng_state": copy.deepcopy(self.rng.get_state())}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            not isinstance(state, Mapping)
            or type(state.get("format_version")) is not int
            or state.get("format_version") != 1
        ):
            raise ValueError("unsupported sequence sampler state")
        rng_state = copy.deepcopy(state.get("rng_state"))
        try:
            probe = np.random.RandomState()
            probe.set_state(rng_state)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid sequence sampler RNG state") from error
        self.rng.set_state(rng_state)

    def _validated_episodes(self) -> tuple[Mapping[str, np.ndarray], ...]:
        source = self._episodes() if callable(self._episodes) else self._episodes
        episodes = tuple(source)
        if not episodes:
            raise ValueError("replay has no episodes")
        action_dim = episodes[0]["action"].shape[1]
        reference_shapes = {
            key: episodes[0][key].shape[1:] for key in REQUIRED_FIELDS
        }
        for episode in episodes:
            validate_episode(episode, action_dim=action_dim)
            for key in REQUIRED_FIELDS:
                if episode[key].shape[1:] != reference_shapes[key]:
                    raise ValueError(f"episode {key} trailing shapes differ")
        if not any(len(episode["reward"]) >= 2 for episode in episodes):
            raise ValueError("replay has no episode with at least two items")
        return episodes

    def _choose_episode(
        self,
        episodes: tuple[Mapping[str, np.ndarray], ...],
        probabilities: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
        index = int(self.rng.choice(len(episodes), p=probabilities))
        return episodes[index]

    def _sample_sequence(
        self,
        episodes: tuple[Mapping[str, np.ndarray], ...],
        probabilities: np.ndarray,
        length: int,
    ) -> dict[str, np.ndarray]:
        remaining = length
        fragments: dict[str, list[np.ndarray]] = {
            key: [] for key in REQUIRED_FIELDS
        }
        first_fragment = True

        while remaining:
            episode = self._choose_episode(episodes, probabilities)
            if len(episode["reward"]) < 2:
                continue
            start = (
                int(self.rng.randint(0, len(episode["reward"]) - 1))
                if first_fragment
                else 0
            )
            stop = min(start + remaining, len(episode["reward"]))
            count = stop - start
            for key in REQUIRED_FIELDS:
                fragment = episode[key][start:stop].copy()
                if key == "is_first":
                    fragment[0] = True
                fragments[key].append(fragment)
            remaining -= count
            first_fragment = False

        return {
            key: np.ascontiguousarray(np.concatenate(parts, axis=0))
            for key, parts in fragments.items()
        }


__all__ = ["SequenceSampler"]
