"""Shared environment construction and episode recording.

``car_racing`` (Box2D), ``dmc`` (mujoco/dm_control), and ``video``
(imageio) are intentionally not re-exported here: each pulls in a
backend-specific dependency that not every consumer needs. Import them
explicitly, e.g. ``from baselines_common.envs.dmc import DMCGymAdapter``.
"""

from .atari import make_atari_env, make_vector_env

__all__ = ["make_atari_env", "make_vector_env"]
