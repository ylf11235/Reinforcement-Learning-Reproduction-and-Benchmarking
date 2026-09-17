"""Shared DMC (dm_control) environment: state-input Gymnasium adapter, no
external shimmy dependency.

The ``dm_control`` import stays lazy (inside ``_build_dmc_raw``) so that
importing this module does not require mujoco; only backends that actually
build DMC environments pay the dependency (see
``baselines_common/envs/__init__.py``).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


def _build_dmc_raw(domain: str, task: str, seed: int):
    """Load a dm_control suite env (seedable via physics rebuild)."""
    from dm_control import suite
    return suite.load(domain_name=domain, task_name=task,
                      task_kwargs={"random": seed})


class DMCGymAdapter(gym.Env):
    """dm_control <-> gymnasium: flattened float obs, Box actions, 1000-step
    episodes, dense raw rewards. One control step (environment transition)
    per env.step; dm_control advances physics substeps internally.

    Determinism: physics is seeded once at load (training seed or eval seed);
    episode resets re-randomize via dm_control's own task RNG. Exact
    reset-seed rebinding is not supported (documented deviation; evaluation
    determinism comes from the action path, not env re-seeding)."""

    metadata = {"render_modes": []}

    def __init__(self, domain: str, task: str, seed: int):
        super().__init__()
        self._domain, self._task = domain, task
        self._env = _build_dmc_raw(domain, task, seed)
        specs = self._env.action_spec()
        self.action_space = gym.spaces.Box(
            low=np.asarray(specs.minimum, dtype=np.float32),
            high=np.asarray(specs.maximum, dtype=np.float32),
            shape=specs.shape, dtype=np.float32,
        )
        obs_dim = self._flat_obs_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self._episode_len = 0

    # -- obs flattening (dict of arrays -> single float32 vector) --
    def _flatten(self, timestep) -> np.ndarray:
        parts = [np.asarray(v, dtype=np.float32).reshape(-1)
                 for v in timestep.observation.values()]
        return np.concatenate(parts)

    def _flat_obs_dim(self) -> int:
        ts = self._env.reset()
        return int(self._flatten(ts).shape[0])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ts = self._env.reset()
        self._episode_len = 0
        return self._flatten(ts), {}

    def step(self, action):
        ts = self._env.step(np.asarray(action, dtype=np.float64))
        self._episode_len += 1
        obs = self._flatten(ts)
        reward = float(ts.reward or 0.0)
        terminated = ts.step_type == ts.step_type.LAST and ts.discount == 0.0
        truncated = (ts.step_type == ts.step_type.LAST
                     and not terminated) or self._episode_len >= 1000
        return obs, reward, bool(terminated), bool(truncated), {
            "raw_reward": reward}

    def render(self):
        """RGB frame from the default tracking camera (for video)."""
        return self._env.physics.render(camera_id=0, height=240, width=320)

    def close(self):
        return None
