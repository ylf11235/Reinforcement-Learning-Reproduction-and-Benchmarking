"""Gymnasium adapter reproducing EADream's pinned Atari preprocessing."""

from __future__ import annotations

from typing import Any, Literal, Mapping

import ale_py
import cv2
import gymnasium as gym
import numpy as np

from .wrappers import StartGuard


AtariMode = Literal[
    "train", "formal_eval", "audit", "upstream_diagnostic", "video"
]
_VALID_MODES = {
    "train",
    "formal_eval",
    "audit",
    "upstream_diagnostic",
    "video",
}
_REGISTERED = False


def _ensure_registration() -> None:
    global _REGISTERED
    if not _REGISTERED:
        gym.register_envs(ale_py)
        _REGISTERED = True


def protocol_for(slug: str, mode: str, configured: str) -> str:
    if mode not in _VALID_MODES:
        raise ValueError(f"unknown Atari environment mode: {mode}")
    if mode == "upstream_diagnostic":
        return "none"
    if slug in {"bowling", "tennis"}:
        return "fire"
    return configured


class EADreamAtariAdapter(gym.Wrapper):
    """Apply raw action repeat, fixed source buffers, and RGB resize."""

    def __init__(
        self,
        env: gym.Env,
        *,
        action_repeat: int = 4,
        size: int = 64,
        noop_max_exclusive: int = 30,
        verify_render_source: bool = False,
    ) -> None:
        super().__init__(env)
        if type(action_repeat) is not int or action_repeat < 2:
            raise ValueError("action_repeat must be an integer of at least two")
        if type(size) is not int or size <= 0:
            raise ValueError("size must be a positive integer")
        if type(noop_max_exclusive) is not int or noop_max_exclusive < 0:
            raise ValueError("noop_max_exclusive must be a non-negative integer")

        raw_space = env.observation_space
        if raw_space.shape != (210, 160, 3) or raw_space.dtype != np.uint8:
            raise ValueError(
                "raw Atari observations must be HWC uint8 with shape (210, 160, 3)"
            )
        self.action_repeat = action_repeat
        self.size = size
        self.noop_max_exclusive = noop_max_exclusive
        self.verify_render_source = verify_render_source
        self._rng = np.random.RandomState()
        self._last_raw = np.zeros(raw_space.shape, dtype=np.uint8)
        self._buffers = np.zeros((2, *raw_space.shape), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    0, 255, shape=(size, size, 3), dtype=np.uint8
                ),
                "is_first": gym.spaces.Discrete(2),
                "is_terminal": gym.spaces.Discrete(2),
            }
        )

    def _action_index(self, meaning: str) -> int:
        meanings = list(self.env.unwrapped.get_action_meanings())
        if meaning not in meanings:
            raise ValueError(f"minimal Atari action set requires {meaning}")
        return meanings.index(meaning)

    def _remember_raw(self, observation: Any) -> None:
        frame = np.asarray(observation)
        if frame.shape != self._last_raw.shape or frame.dtype != np.uint8:
            raise ValueError(
                "raw Atari observations must remain HWC uint8 with shape "
                "(210, 160, 3)"
            )
        np.copyto(self._last_raw, frame)

    def _capture_screen(self, destination: np.ndarray) -> None:
        np.copyto(destination, self._last_raw)

    def resize(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            image, (self.size, self.size), interpolation=cv2.INTER_AREA
        )
        return np.ascontiguousarray(resized, dtype=np.uint8)

    def _observation(
        self, *, is_first: bool, is_terminal: bool
    ) -> dict[str, object]:
        pooled = np.maximum(self._buffers[0], self._buffers[1])
        return {
            "image": self.resize(pooled),
            "is_first": bool(is_first),
            "is_terminal": bool(is_terminal),
        }

    def _current_lives(self) -> int | None:
        ale = getattr(self.env.unwrapped, "ale", None)
        if ale is None:
            return None
        try:
            return int(ale.lives())
        except (AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def _termination_reason(terminated: bool, truncated: bool) -> str | None:
        if terminated:
            return "terminated"
        if truncated:
            return "truncated"
        return None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        raw, raw_info = self.env.reset(seed=seed, options=options)
        self._remember_raw(raw)
        reset_lives = self._current_lives()
        noop_frames = (
            int(self._rng.randint(self.noop_max_exclusive))
            if self.noop_max_exclusive
            else 0
        )
        noop_reward = 0.0
        noop_action = self._action_index("NOOP") if noop_frames else 0
        final_info = dict(raw_info)
        for _ in range(noop_frames):
            raw, reward, terminated, truncated, step_info = self.env.step(noop_action)
            self._remember_raw(raw)
            noop_reward += float(reward)
            final_info.update(step_info)
            if terminated or truncated:
                raw, restarted_info = self.env.reset(seed=None, options=options)
                self._remember_raw(raw)
                final_info.update(restarted_info)

        self._capture_screen(self._buffers[0])
        self._buffers[1].fill(0)
        lives_after = self._current_lives()
        info = dict(final_info)
        info.update(
            {
                "raw_reward": 0.0,
                "policy_reward": 0.0,
                "reset_reward": noop_reward,
                "reset_noop_reward": noop_reward,
                "forced_start_reward": 0.0,
                "forced_reset_start_reward": 0.0,
                "forced_life_start_reward": 0.0,
                "lives_before": reset_lives,
                "lives_after": lives_after,
                "termination_reason": None,
                "agent_interactions": 0,
                "ale_frames": 0,
                "reset_noop_frames": noop_frames,
                "forced_start_actions": 0,
                "forced_start_frames": 0,
                "forced_reset_start_actions": 0,
                "forced_reset_start_frames": 0,
                "forced_life_start_actions": 0,
                "forced_life_start_frames": 0,
            }
        )
        if self.verify_render_source:
            self._verified_native_render()
        return self._observation(is_first=True, is_terminal=False), info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid minimal Atari action: {action!r}")
        lives_before = self._current_lives()
        total_reward = 0.0
        ale_frames = 0
        terminated = False
        truncated = False
        final_info: dict[str, Any] = {}
        for repeat in range(self.action_repeat):
            raw, reward, terminated, truncated, raw_info = self.env.step(int(action))
            self._remember_raw(raw)
            total_reward += float(reward)
            ale_frames += 1
            final_info = dict(raw_info)
            if repeat == self.action_repeat - 2:
                self._capture_screen(self._buffers[1])
            if terminated or truncated:
                break
        self._capture_screen(self._buffers[0])

        lives_after = self._current_lives()
        info = dict(final_info)
        info.update(
            {
                "raw_reward": total_reward,
                "policy_reward": total_reward,
                "reset_reward": 0.0,
                "reset_noop_reward": 0.0,
                "forced_start_reward": 0.0,
                "forced_reset_start_reward": 0.0,
                "forced_life_start_reward": 0.0,
                "lives_before": lives_before,
                "lives_after": lives_after,
                "termination_reason": self._termination_reason(
                    terminated, truncated
                ),
                "agent_interactions": 1,
                "ale_frames": ale_frames,
                "reset_noop_frames": 0,
                "forced_start_actions": 0,
                "forced_start_frames": 0,
                "forced_reset_start_actions": 0,
                "forced_reset_start_frames": 0,
                "forced_life_start_actions": 0,
                "forced_life_start_frames": 0,
            }
        )
        if self.verify_render_source:
            self._verified_native_render()
        terminal = terminated or truncated
        return (
            self._observation(is_first=False, is_terminal=terminal),
            total_reward,
            terminated,
            truncated,
            info,
        )

    def _verified_native_render(self) -> np.ndarray:
        rendered = np.asarray(self.env.render())
        if rendered.shape != self._last_raw.shape or rendered.dtype != np.uint8:
            raise ValueError(
                "native rgb_array render shape/dtype differs from raw observation"
            )
        return np.ascontiguousarray(rendered)

    def render(self) -> np.ndarray:
        if self.verify_render_source:
            return self._verified_native_render().copy()
        return self._last_raw.copy()


def make_atari_env(
    config: Mapping[str, Any],
    *,
    mode: AtariMode,
) -> gym.Env:
    """Build an unreset Atari environment from one resolved Task 1 config."""
    if mode not in _VALID_MODES:
        raise ValueError(f"unknown Atari environment mode: {mode}")
    _ensure_registration()
    game = config["game"]
    environment = config["environment"]
    make_kwargs: dict[str, Any] = {
        "obs_type": "rgb",
        "frameskip": 1,
        "repeat_action_probability": 0.0,
        "full_action_space": False,
        "max_num_frames_per_episode": 108_000,
    }
    if mode == "video":
        make_kwargs["render_mode"] = "rgb_array"
    raw_env = gym.make(game["env_id"], **make_kwargs)
    adapter = EADreamAtariAdapter(
        raw_env,
        action_repeat=environment["action_repeat"],
        size=environment["screen_size"],
        noop_max_exclusive=environment["noop_max_exclusive"],
        verify_render_source=mode == "video",
    )
    protocol = protocol_for(
        game["slug"], mode, environment["start_protocol"]
    )
    return StartGuard(adapter, protocol=protocol)
