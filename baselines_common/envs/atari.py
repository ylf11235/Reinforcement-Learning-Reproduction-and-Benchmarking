"""Atari preprocessing shared by the SAC and SB3 PPO baselines.

The wrapper order intentionally mirrors ``DeepCriticEngine/deepcritic/envs.py``:
ALE frameskip is disabled, then Gymnasium's AtariPreprocessing performs action
repeat, max-pooling, grayscale conversion and resize, followed by a four-frame
stack.  The returned observation is channel-first uint8.
"""

from __future__ import annotations

from typing import Callable, Literal

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


_REGISTERED = False


class SignReward(gym.RewardWrapper):
    """Clip Atari rewards to their sign while preserving termination flags."""

    def reward(self, reward):
        return float(np.sign(reward))


StartProtocol = Literal["none", "fire", "move"]


class RawRewardInfo(gym.Wrapper):
    """Expose the pre-clipping reward in step info for diagnostics and audits."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        result_info = dict(info)
        result_info["raw_reward"] = float(reward)
        return observation, reward, terminated, truncated, result_info


class StartProtocolWrapper(gym.Wrapper):
    """Start games that wait for FIRE or a movement action after reset/life loss."""

    _VALID_PROTOCOLS = {"none", "fire", "move"}

    def __init__(self, env: gym.Env, protocol: StartProtocol = "none") -> None:
        super().__init__(env)
        if protocol not in self._VALID_PROTOCOLS:
            raise ValueError(f"unknown Atari start protocol: {protocol}")
        self.protocol = protocol
        self._lives: int | None = None

    def _start_action(self) -> int | None:
        if self.protocol == "none":
            return None

        meanings = list(self.env.unwrapped.get_action_meanings())
        if self.protocol == "fire":
            if "FIRE" not in meanings:
                raise ValueError("FIRE start protocol requires a FIRE action")
            return meanings.index("FIRE")

        for movement in ("RIGHT", "LEFT", "UP", "DOWN"):
            if movement in meanings:
                return meanings.index(movement)
        raise ValueError("move start protocol requires a cardinal movement action")

    def _current_lives(self) -> int | None:
        ale = getattr(self.env.unwrapped, "ale", None)
        if ale is None:
            return None
        try:
            return int(ale.lives())
        except (AttributeError, TypeError):
            return None

    def _apply_start_action(self):
        action = self._start_action()
        if action is None:
            return None
        return self.env.step(action)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        started = self._apply_start_action()
        if started is not None:
            observation, _, terminated, truncated, start_info = started
            if terminated or truncated:
                observation, info = self.env.reset(seed=seed, options=options)
            else:
                info = dict(info)
                info.update(start_info)
                info["start_protocol"] = self.protocol
        self._lives = self._current_lives()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        result_info = dict(info)
        forced_start_actions = 0
        current_lives = self._current_lives()

        if (
            not (terminated or truncated)
            and self._lives is not None
            and current_lives is not None
            and 0 < current_lives < self._lives
        ):
            started = self._apply_start_action()
            if started is not None:
                observation, start_reward, terminated, truncated, start_info = started
                reward = float(reward) + float(start_reward)
                result_info.update(start_info)
                forced_start_actions = 1

        self._lives = self._current_lives()
        if forced_start_actions:
            result_info["forced_start_actions"] = forced_start_actions
            result_info["start_protocol"] = self.protocol
        return observation, reward, terminated, truncated, result_info


def _ensure_registration() -> None:
    global _REGISTERED
    if not _REGISTERED:
        gym.register_envs(ale_py)
        _REGISTERED = True


def make_atari_env(
    env_id: str = "ALE/Pong-v5",
    seed: int = 0,
    *,
    frame_skip: int = 4,
    screen_size: int = 84,
    stack: int = 4,
    noop_max: int = 30,
    clip_reward: bool = True,
    repeat_action_probability: float = 0.0,
    max_num_frames_per_episode: int = 108_000,
    start_protocol: StartProtocol = "none",
    render_mode: str | None = None,
) -> gym.Env:
    """Build one deterministic Atari environment with the common contract."""
    _ensure_registration()
    make_kwargs = {
        "frameskip": 1,
        "repeat_action_probability": repeat_action_probability,
        "max_num_frames_per_episode": max_num_frames_per_episode,
        "obs_type": "rgb",
    }
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode
    env = gym.make(env_id, **make_kwargs)
    env = AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = StartProtocolWrapper(env, protocol=start_protocol)
    env = RawRewardInfo(env)
    if clip_reward:
        env = SignReward(env)
    env = FrameStackObservation(env, stack_size=stack)
    obs, _ = env.reset(seed=seed)
    expected = spaces.Box(
        low=0,
        high=255,
        shape=(stack, screen_size, screen_size),
        dtype=np.uint8,
    )
    if env.observation_space.shape != expected.shape:
        raise ValueError(
            f"unexpected Atari stack shape: {env.observation_space.shape}; "
            f"expected {expected.shape}"
        )
    if np.asarray(obs).dtype != np.uint8:
        raise TypeError(f"unexpected Atari dtype: {np.asarray(obs).dtype}")
    return env


def make_vector_env(
    env_id: str = "ALE/Pong-v5",
    num_envs: int = 8,
    seed: int = 0,
    **kwargs,
) -> gym.vector.SyncVectorEnv:
    """Build synchronous environments with stable, non-overlapping seeds."""
    factories: list[Callable[[], gym.Env]] = []
    seed_sequences = np.random.SeedSequence(int(seed)).spawn(int(num_envs))
    for seed_sequence in seed_sequences:
        worker_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
        factories.append(
            lambda worker_seed=worker_seed: make_atari_env(
                env_id, worker_seed, **kwargs
            )
        )
    return gym.vector.SyncVectorEnv(
        factories,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def extract_final_observation(infos, observations, index: int):
    """Read SAME_STEP's final observation for a terminated vector slot."""
    final_obs = infos.get("final_obs") if isinstance(infos, dict) else None
    final_mask = infos.get("_final_obs") if isinstance(infos, dict) else None
    if final_obs is None or (final_mask is not None and not bool(final_mask[index])):
        return observations[index]
    value = final_obs[index]
    return observations[index] if value is None else value
