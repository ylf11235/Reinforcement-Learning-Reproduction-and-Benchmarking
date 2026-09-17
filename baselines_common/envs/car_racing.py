"""Shared CarRacing environment: factory + adapter (pixel obs, continuous
actions). House benchmark contract: BT.601 grayscale CHW uint8 frames,
configurable frame stack, action repeat with summed rewards, [-1,1]^3
actions rescaled to the native Box.

Environment classes live in baselines_common regardless of how many
backends currently consume them (DevRequirement.md): benchmark
environments are inherently intended for evaluation by multiple
algorithms.
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np

CAR_RACING_ENV_KWARGS = {
    "continuous": True,
    "domain_randomize": False,
    "lap_complete_percent": 0.95,
}


def make_car_racing_env():
    return gym.make("CarRacing-v3", **CAR_RACING_ENV_KWARGS)


class CarRacingAdapter(gym.Env):
    """CarRacing-v3 -> shared pixel+continuous benchmark contract.

    - Observation: BT.601 integer luma (299R+587G+114B+500)//1000, uint8,
      CHW (1,H,W); returns the last ``frame_stack`` frames stacked oldest
      first, newest last (callers storing only the newest frame can
      reconstruct the stack).
    - Action: exposes Box(-1,1,(3,)); rescales to the native
      Box([-1,0,0],[1,1,1]) via low + (a+1)/2 * (high-low).
    - Action repeat 2: same action applied ``action_repeat`` env steps,
      rewards summed, episode flags OR-ed, repeat stops early on end.
    - Episode cap: ``max_frames`` raw env frames (1000 default; native
      truncation may fire earlier).
    - Seeding: a constructor ``seed`` is applied to the first ``reset()``
      that does not pass one (and to action_space), matching the
      seeded-at-construction behavior of the DMC adapter.
    """

    metadata = {"render_modes": []}

    def __init__(self, env, *, frame_stack: int = 4, action_repeat: int = 2,
                 max_frames: int = 1000, seed: int | None = None):
        self._env = env
        self._stack, self._repeat, self._max_frames = (
            int(frame_stack), int(action_repeat), int(max_frames))
        h, w, _ = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            0, 255, (self._stack, h, w), dtype=np.uint8)
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32)
        self._low = np.asarray(env.action_space.low, dtype=np.float32)
        self._high = np.asarray(env.action_space.high, dtype=np.float32)
        self._frames = deque(maxlen=self._stack)
        self._frames_in_episode = 0
        self._pending_seed = seed
        if seed is not None:
            self.action_space.seed(seed)

    @staticmethod
    def _to_gray(obs) -> np.ndarray:
        arr = np.asarray(obs)
        r = arr[..., 0].astype(np.uint32)
        g = arr[..., 1].astype(np.uint32)
        b = arr[..., 2].astype(np.uint32)
        gray = ((299 * r + 587 * g + 114 * b + 500) // 1000).astype(np.uint8)
        return gray[np.newaxis]  # (1, H, W)

    def _stacked(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0)

    def reset(self, *, seed=None, options=None):
        if seed is None and self._pending_seed is not None:
            seed, self._pending_seed = self._pending_seed, None
        super().reset(seed=seed)
        obs, info = self._env.reset(seed=seed, options=options)
        frame = self._to_gray(obs)
        self._frames = deque([frame] * self._stack, maxlen=self._stack)
        self._frames_in_episode = 0
        return self._stacked(), info

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        native = self._low + (a + 1.0) * 0.5 * (self._high - self._low)
        total = 0.0
        term = trunc = False
        obs = None
        info = {}
        for _ in range(self._repeat):
            obs, r, term, trunc, info = self._env.step(native)
            total += float(r)
            self._frames_in_episode += 1
            if term or trunc:
                break
        trunc = bool(trunc or self._frames_in_episode >= self._max_frames)
        self._frames.append(self._to_gray(obs))
        return self._stacked(), total, bool(term), trunc, {
            "raw_reward": total}
