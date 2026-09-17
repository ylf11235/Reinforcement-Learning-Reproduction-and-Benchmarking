"""Evaluation loops for MRQ: periodic eval and final audit (deterministic)."""
from __future__ import annotations

import numpy as np


def run_episode(agent, env, eval_eps_random: float = 0.0, rng=None):
    """One deterministic episode. Returns (raw_return, steps).

    ``eval_eps_random`` (regression variant only): with prob eps replace the
    action with a uniform sample — applied AFTER deterministic selection,
    per PROTOCOL.md decision 1.
    """
    obs, _ = env.reset()
    raw_return = 0.0
    steps = 0
    done = False
    while not done:
        action = agent.select_action(obs, evaluate=True)
        if action is None:  # warmup-sized buffer: fall back to random
            action = env.action_space.sample()
        if eval_eps_random > 0 and rng is not None and rng.random() < eval_eps_random:
            action = int(env.action_space.sample())
        obs, reward, term, trunc, info = env.step(action)
        raw = info.get("raw_reward", reward)
        raw_return += float(raw)
        steps += 1
        done = term or trunc
    return raw_return, steps


def periodic_eval(agent, cfg, eval_seed_base: int):
    """10 deterministic episodes; returns mean raw return and per-episode list."""
    import env_adapter
    ev = cfg["evaluation"]
    eps_random = ev.get("eval_eps_random", 0.0) if cfg["experiment"]["variant"] == "regression" else 0.0
    returns = []
    for i in range(ev["periodic_episodes"]):
        seed = ev["periodic_seed_base"] + i
        env = env_adapter.build_eval_env(cfg, eval_seed=seed)
        rng = np.random.default_rng(seed) if eps_random > 0 else None
        r, _ = run_episode(agent, env, eps_random, rng)
        returns.append(r)
        env.close()
    return float(np.mean(returns)), returns


def final_audit(agent, cfg):
    """30 deterministic episodes on the final weights; raw unclipped returns."""
    import env_adapter
    ev = cfg["evaluation"]
    eps_random = ev.get("eval_eps_random", 0.0) if cfg["experiment"]["variant"] == "regression" else 0.0
    returns = []
    for i in range(ev["audit_episodes"]):
        seed = ev["audit_seed_base"] + i
        env = env_adapter.build_eval_env(cfg, eval_seed=seed)
        rng = np.random.default_rng(seed) if eps_random > 0 else None
        r, _ = run_episode(agent, env, eps_random, rng)
        returns.append(r)
        env.close()
    return returns
