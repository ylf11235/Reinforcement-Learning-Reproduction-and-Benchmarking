"""Shared Atari environment construction and episode recording.

``video`` is intentionally not re-exported here: it pulls in ``imageio``,
which not every backend needs. Import it explicitly as
``from baselines_common.envs.video import ...``.
"""

from .atari import make_atari_env, make_vector_env

__all__ = ["make_atari_env", "make_vector_env"]
