"""EADream-specific Atari episode protocol wrappers."""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np

from eadream.events.mog2 import MOG2Config, MOG2EventExtractor


StartProtocol = Literal["none", "fire"]


class EventObservationWrapper(gym.Wrapper):
    """Add one stateful MOG2 event mask to each structured observation."""

    def __init__(self, env: gym.Env, config: MOG2Config | None = None) -> None:
        super().__init__(env)
        self.config = MOG2Config() if config is None else config
        self._extractor: MOG2EventExtractor | None = None
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError("EventObservationWrapper requires a Dict observation space")
        spaces = dict(env.observation_space.spaces)
        image_space = spaces.get("image")
        if image_space is None or image_space.shape != (64, 64, 3):
            raise ValueError("EventObservationWrapper requires a 64x64 RGB image")
        spaces["event"] = gym.spaces.Box(0, 255, (64, 64), np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)

    @staticmethod
    def _with_event(observation: Any, event: np.ndarray) -> dict[str, object]:
        if type(observation) is not dict:
            raise TypeError("EventObservationWrapper requires plain dict observations")
        result = dict(observation)
        result["event"] = event
        return result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        observation, info = self.env.reset(seed=seed, options=options)
        self._extractor = MOG2EventExtractor(self.config)
        event = self._extractor.reset(observation["image"])
        return self._with_event(observation, event), info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self._extractor is None:
            raise RuntimeError("reset must be called before step")
        event = self._extractor.step(observation["image"])
        return self._with_event(observation, event), reward, terminated, truncated, info


class StartGuard(gym.Wrapper):
    """Insert approved FIRE decisions after reset and nonterminal life loss."""

    def __init__(self, env: gym.Env, protocol: StartProtocol = "none") -> None:
        super().__init__(env)
        if protocol not in {"none", "fire"}:
            raise ValueError(f"unknown Atari start protocol: {protocol}")
        self.protocol = protocol
        self._start_action = self._resolve_start_action()
        self._lives: int | None = None

    def _resolve_start_action(self) -> int | None:
        if self.protocol == "none":
            return None
        meanings = list(self.env.unwrapped.get_action_meanings())
        if "FIRE" not in meanings:
            raise ValueError("FIRE start protocol requires a FIRE action")
        return meanings.index("FIRE")

    def _current_lives(self) -> int | None:
        ale = getattr(self.env.unwrapped, "ale", None)
        if ale is None:
            return None
        try:
            return int(ale.lives())
        except (AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def _as_reset_observation(observation: Any) -> Any:
        if not isinstance(observation, dict):
            return observation
        result = dict(observation)
        result["is_first"] = True
        result["is_terminal"] = False
        return result

    @staticmethod
    def _as_step_observation(
        observation: Any, terminated: bool, truncated: bool
    ) -> Any:
        if not isinstance(observation, dict):
            return observation
        result = dict(observation)
        result["is_first"] = False
        result["is_terminal"] = bool(terminated or truncated)
        return result

    @staticmethod
    def _zero_start_fields(info: dict[str, Any]) -> None:
        info.setdefault("forced_start_reward", 0.0)
        info.setdefault("forced_start_actions", 0)
        info.setdefault("forced_start_frames", 0)
        info.setdefault("forced_reset_start_reward", 0.0)
        info.setdefault("forced_reset_start_actions", 0)
        info.setdefault("forced_reset_start_frames", 0)
        info.setdefault("forced_life_start_reward", 0.0)
        info.setdefault("forced_life_start_actions", 0)
        info.setdefault("forced_life_start_frames", 0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        observation, reset_info = self.env.reset(seed=seed, options=options)
        result_info = dict(reset_info)
        self._zero_start_fields(result_info)
        result_info["start_protocol"] = self.protocol
        if self._start_action is None:
            self._lives = result_info.get("lives_after", self._current_lives())
            return observation, result_info

        started, start_reward, terminated, truncated, start_info = self.env.step(
            self._start_action
        )
        start_frames = int(start_info.get("ale_frames", 0))
        aborted = bool(terminated or truncated)
        first_noop_frames = int(result_info.get("reset_noop_frames", 0))
        first_noop_reward = float(result_info.get("reset_noop_reward", 0.0))
        if aborted:
            observation, restarted_info = self.env.reset(seed=seed, options=options)
            result_info = dict(restarted_info)
            result_info["reset_noop_frames"] = first_noop_frames + int(
                result_info.get("reset_noop_frames", 0)
            )
            result_info["reset_noop_reward"] = first_noop_reward + float(
                result_info.get("reset_noop_reward", 0.0)
            )
            result_info["reset_reward"] = result_info["reset_noop_reward"]
        else:
            observation = started
            result_info["lives_after"] = start_info.get(
                "lives_after", result_info.get("lives_after")
            )

        self._zero_start_fields(result_info)
        result_info.update(
            {
                "raw_reward": 0.0,
                "policy_reward": 0.0,
                "forced_start_reward": float(start_reward),
                "forced_start_actions": 1,
                "forced_start_frames": start_frames,
                "forced_reset_start_reward": float(start_reward),
                "forced_reset_start_actions": 1,
                "forced_reset_start_frames": start_frames,
                "forced_reset_start_aborted": aborted,
                "agent_interactions": 0,
                "ale_frames": 0,
                "start_protocol": self.protocol,
            }
        )
        self._lives = result_info.get("lives_after", self._current_lives())
        return self._as_reset_observation(observation), result_info

    def step(self, action: int):
        observation, reward, terminated, truncated, action_info = self.env.step(
            action
        )
        result_info = dict(action_info)
        self._zero_start_fields(result_info)
        result_info["start_protocol"] = self.protocol
        policy_reward = float(reward)
        policy_frames = int(result_info.get("ale_frames", 0))
        lives_before = result_info.get("lives_before", self._lives)
        lives_after = result_info.get("lives_after", self._current_lives())

        life_lost = (
            not (terminated or truncated)
            and self._start_action is not None
            and lives_before is not None
            and lives_after is not None
            and 0 < int(lives_after) < int(lives_before)
        )
        if life_lost:
            started, start_reward, terminated, truncated, start_info = self.env.step(
                self._start_action
            )
            observation = started
            reward = policy_reward + float(start_reward)
            start_frames = int(start_info.get("ale_frames", 0))
            result_info.update(
                {
                    "raw_reward": float(reward),
                    "policy_reward": policy_reward,
                    "forced_start_reward": float(start_reward),
                    "forced_start_actions": 1,
                    "forced_start_frames": start_frames,
                    "forced_life_start_reward": float(start_reward),
                    "forced_life_start_actions": 1,
                    "forced_life_start_frames": start_frames,
                    "lives_before": lives_before,
                    "lives_after": start_info.get("lives_after", lives_after),
                    "termination_reason": start_info.get("termination_reason"),
                    "agent_interactions": 1,
                    "ale_frames": policy_frames,
                    "start_protocol": self.protocol,
                }
            )
        else:
            reward = policy_reward
            result_info["raw_reward"] = policy_reward
            result_info["policy_reward"] = policy_reward

        self._lives = result_info.get("lives_after", self._current_lives())
        observation = self._as_step_observation(
            observation, terminated, truncated
        )
        return observation, reward, terminated, truncated, result_info
