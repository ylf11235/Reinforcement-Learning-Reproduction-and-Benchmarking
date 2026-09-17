"""MRQ experiment CLI: validate / preflight / run / campaign / status /
summarize / archive.

Lifecycle per PROTOCOL.md §8 and the plan Task 7 behavior spec:
- atomic JSON writes; hashed checkpoint generations with manifests;
- run_state: PENDING|RUNNING|COMPLETED|FAILED|INTERRUPTED;
- COMPLETED only after final checkpoint + audit + scores + provenance and
  their hashes validate;
- resume: REJECTED (model_only; PROTOCOL decision 2);
- campaign: skips only hash-verified completed runs; records failures;
- archive: validates then refuses overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

MRQ_DIR = Path(__file__).resolve().parent
REPO_ROOT = MRQ_DIR.parents[1]  # algorithms/mrq -> algorithms -> repo root
if str(MRQ_DIR) not in sys.path:
    sys.path.insert(0, str(MRQ_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config, config_fingerprint, ConfigError  # noqa: E402
import env_adapter  # noqa: E402
import evaluation as eval_lib  # noqa: E402

_FAKE_ENV = False  # tests flip this to swap in the registered fake env

RUN_STATES = ("PENDING", "RUNNING", "COMPLETED", "FAILED", "INTERRUPTED")


def _frame_mult(cfg: dict) -> int:
    """Raw env frames per agent step: DMC 1 (no repeat), car_racing 2
    (action repeat 2), Atari 4 (wrapper action-repeat 4)."""
    return {"dmc": 1, "car_racing": 2}.get(cfg["experiment"]["variant"], 4)


def _resolve_games_dir(campaign_cfg_path: Path, campaign_id: str) -> Path:
    """Per-game config lookup for suite campaigns. The games/ directory may
    sit next to the suite file (suite manifest inside the campaign dir) or,
    for the generator/DMC convention, beside the top-level manifest
    configs/<campaign>.yaml -> configs/<campaign>/games/."""
    candidates = [
        campaign_cfg_path.parent / "games",
        MRQ_DIR / "configs" / campaign_id / "games",
        campaign_cfg_path.parent,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return campaign_cfg_path.parent


# ------------------------------------------------------------------ helpers

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _write_json(path: Path, obj) -> None:
    _atomic_write(Path(path), json.dumps(obj, indent=2, sort_keys=False))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_tree(folder: Path) -> dict:
    return {
        p.relative_to(folder).as_posix(): _sha256_file(p)
        for p in sorted(folder.rglob("*")) if p.is_file()
    }


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_checkpoint_generation(agent, gen_dir: Path, step: int,
                                fingerprint: str, durable: bool):
    """Write a checkpoint generation atomically with a hash manifest."""
    gen_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".gen_", dir=str(gen_dir.parent)))
    try:
        agent.save_checkpoint(str(tmp), durable=durable)
        manifest = {
            "step": step,
            "config_fingerprint": fingerprint,
            "durable": durable,
            "created_at": _now_iso(),
            "files": {
                p.name: _sha256_file(p) for p in sorted(tmp.iterdir())
                if p.is_file()
            },
        }
        _write_json(tmp / "manifest.json", manifest)
        if gen_dir.exists():
            shutil.rmtree(gen_dir)
        os.replace(tmp, gen_dir)
    finally:
        if tmp.exists() and gen_dir != tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _verify_generation(gen_dir: Path, fingerprint: str) -> bool:
    manifest_path = gen_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config_fingerprint") != fingerprint:
        return False
    for name, digest in manifest.get("files", {}).items():
        f = gen_dir / name
        if not f.exists() or _sha256_file(f) != digest:
            return False
    return True


def _verify_completed_run(run_dir: Path, fingerprint: str) -> bool:
    """Full completed-run gate: state, fingerprint, summary, provenance, and
    the final checkpoint generation must all be present and consistent
    before a run may be skipped by a campaign or accepted by archive."""
    try:
        state = json.loads((run_dir / "run_state.json").read_text())
        scores = json.loads((run_dir / "summary" / "scores.json").read_text())
        prov = json.loads((run_dir / "provenance.json").read_text())
    except (OSError, ValueError):
        return False
    if state.get("state") != "COMPLETED":
        return False
    if state.get("fingerprint") != fingerprint:
        return False
    if prov.get("config_fingerprint") != fingerprint:
        return False
    if scores.get("mean_raw_return") != state.get("mean_raw_return"):
        return False
    return _verify_generation(run_dir / "checkpoints" / "final", fingerprint)


# ------------------------------------------------------------------- preflight

def _cmd_preflight(args) -> int:
    cfg = load_config(args.config)
    issues = []
    if _FAKE_ENV:
        print("[preflight] fake-env mode: skipping ROM/CUDA checks")
    else:
        try:
            env = env_adapter.build_eval_env(cfg, eval_seed=cfg["runtime"]["seed"] + 100)
            env.reset(); env.close()
        except Exception as e:  # noqa: BLE001
            issues.append(f"env construction failed: {e}")
        if torch.cuda.is_available():
            free_vram, total_vram = torch.cuda.mem_get_info()
            print(f"[preflight] GPU VRAM free {free_vram / 2**30:.1f} GB "
                  f"of {total_vram / 2**30:.1f} GB")
        if cfg["runtime"].get("require_device") and not torch.cuda.is_available():
            issues.append("require_device=true but CUDA unavailable")
    free = shutil.disk_usage(str(MRQ_DIR)).free
    if free < 20 * 2**30:
        issues.append(f"low disk: {free/2**30:.1f} GB free")
    if issues:
        for i in issues:
            print(f"[preflight] FAIL {i}")
        return 1
    print(f"[preflight] OK disk={free/2**30:.0f}GB "
          f"variant={cfg['experiment']['variant']} "
          f"steps={cfg['budget']['steps_requested']:,}")
    return 0


# ----------------------------------------------------------------------- run

def _apply_smoke(cfg: dict) -> dict:
    """Canonical smoke overrides — one source of truth so run and campaign
    skip-verification compute identical fingerprints."""
    cfg["budget"]["steps_requested"] = 800
    cfg["budget"]["raw_frames"] = 800 * _frame_mult(cfg)
    cfg["evaluation"]["periodic_freq"] = 300
    cfg["evaluation"]["audit_episodes"] = 2
    cfg["checkpoint"]["durable_every"] = 800
    cfg["algorithm"]["buffer_size_before_training"] = min(
        50, cfg["algorithm"]["buffer_size_before_training"])
    cfg["algorithm"]["target_update_freq"] = 4
    cfg["algorithm"]["encoder_burst_steps"] = 4
    return cfg


def _run_one(cfg_path: str, run_dir: Path, smoke: bool, resume: bool) -> int:
    cfg = load_config(cfg_path)
    if smoke:
        _apply_smoke(cfg)
    fingerprint = config_fingerprint(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    # input + resolved config
    src = Path(cfg_path).read_text(encoding="utf-8")
    _atomic_write(run_dir / "input_config.yaml", src)
    _atomic_write(run_dir / "resolved_config.yaml",
                  yaml.safe_dump(cfg, sort_keys=False))
    _write_json(run_dir / "run_state.json",
                {"state": "RUNNING", "started_at": _now_iso(),
                 "fingerprint": fingerprint})

    if resume:
        _write_json(run_dir / "run_state.json",
                    {"state": "INTERRUPTED",
                     "reason": "resume rejected: model_only checkpoints "
                               "cannot exactly continue (PROTOCOL decision 2)",
                     "fingerprint": fingerprint})
        print("resume rejected: fresh run required", file=sys.stderr)
        return 2

    seed = cfg["runtime"]["seed"]
    _set_seed(seed)
    # performance flags from the resolved config (semantic-neutral)
    torch.backends.cudnn.benchmark = bool(cfg["runtime"].get("cudnn_benchmark", True))
    # fake-env test path always runs on CPU (test isolation, small VRAM)
    device_name = cfg["runtime"]["device"]
    if _FAKE_ENV or device_name == "cpu":
        device = torch.device("cpu")
    elif device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if _FAKE_ENV:
        import gymnasium as gym
        env = gym.make(cfg["game"]["env_id"])
    else:
        env = env_adapter.build_training_env(cfg)

    from agent import MRQAgent
    obs_shape = env.observation_space.shape
    # observation is the stacked (H, ...) state; the buffer stores single
    # frames (history reassembly happens at sample time): Atari/CarRacing
    # grayscale frame = (1, h, w); DMC vector obs as-is
    if len(obs_shape) == 3:
        frame_shape = (1, obs_shape[1], obs_shape[2])
    else:
        frame_shape = obs_shape
    act_space = env.action_space
    if isinstance(act_space, __import__("gymnasium").spaces.Box):
        action_dim, discrete = int(act_space.shape[0]), False
    else:
        action_dim, discrete = int(act_space.n), True
    agent = MRQAgent(frame_shape, action_dim, discrete=discrete,
                     device=device, history=cfg["environment"]["stack"],
                     algorithm=cfg["algorithm"])

    eval_override = dict(cfg)
    if _FAKE_ENV:
        # keep eval on the fake env too
        pass

    steps_total = cfg["budget"]["steps_requested"]
    periodic_freq = cfg["evaluation"]["periodic_freq"]
    durable_every = cfg["checkpoint"]["durable_every"]
    warmup = cfg["algorithm"]["buffer_size_before_training"]
    horizon = max(cfg["algorithm"]["enc_horizon"],
                  cfg["algorithm"]["q_horizon"])

    train_log = open(run_dir / "train_log.jsonl", "a", encoding="utf-8")
    tb_dir = run_dir / "tensorboard"
    tb_dir.mkdir(exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(str(tb_dir))

    eval_hook = _fake_eval_env if _FAKE_ENV else env_adapter.build_eval_env

    obs, _ = env.reset(seed=seed)
    ep_raw, ep_clip, ep_len = 0.0, 0.0, 0
    t_start = time.perf_counter()
    eval_seconds = 0.0
    ckpt_seconds = 0.0
    steps_done = 0

    def periodic(step):
        nonlocal eval_seconds
        t0 = time.perf_counter()
        mean_r, per_ep = _run_periodic(agent, cfg, eval_hook)
        eval_seconds += time.perf_counter() - t0
        tb.add_scalar("eval/periodic_mean_raw", mean_r, step)
        train_log.write(json.dumps({"step": step, "periodic_eval_mean": mean_r,
                                    "episodes": per_ep}) + "\n")
        train_log.flush()

    try:
        while steps_done < steps_total:
            action = agent.select_action(obs, evaluate=False)
            if action is None:
                action = env.action_space.sample()
            next_obs, r, term, trunc, info = env.step(action)
            raw = float(info.get("raw_reward", r))
            agent.observe(obs, action, next_obs, float(r), term, trunc)
            obs = next_obs
            ep_raw += raw
            ep_clip += float(r)
            ep_len += 1
            steps_done += 1

            if steps_done > warmup and agent.replay_buffer.can_sample(horizon):
                logs = agent.train_step()
                if logs and steps_done % 100 == 0:
                    fps = steps_done / (time.perf_counter() - t_start)
                    rec = {"step": steps_done, "fps": round(fps, 1),
                           "ep_raw": ep_raw, "ep_clip": ep_clip, "ep_len": ep_len}
                    tb.add_scalar("time/fps", fps, steps_done)
                    tb.add_scalar("rollout/ep_raw_rew", ep_raw, steps_done)
                    train_log.write(json.dumps(rec) + "\n")
                    train_log.flush()

            if term or trunc:
                obs, _ = env.reset()
                ep_raw, ep_clip, ep_len = 0.0, 0.0, 0

            if steps_done % periodic_freq == 0:
                periodic(steps_done)
                t0 = time.perf_counter()
                _save_checkpoint_generation(
                    agent, run_dir / "checkpoints" / f"model_{steps_done:010d}",
                    steps_done, fingerprint, durable=False)
                ckpt_seconds += time.perf_counter() - t0
            if steps_done % durable_every == 0:
                t0 = time.perf_counter()
                _save_checkpoint_generation(
                    agent, run_dir / "checkpoints" / f"durable_{steps_done:010d}",
                    steps_done, fingerprint, durable=True)
                ckpt_seconds += time.perf_counter() - t0

        # final checkpoint FIRST, then audit (plan spec)
        _save_checkpoint_generation(
            agent, run_dir / "checkpoints" / "final",
            steps_done, fingerprint, durable=False)

        t0 = time.perf_counter()
        audit_returns = _run_audit(agent, cfg, eval_hook)
        eval_seconds += time.perf_counter() - t0

        scores = {
            "game": cfg["game"]["slug"],
            "variant": cfg["experiment"]["variant"],
            "steps_requested": steps_total,
            "raw_frames": cfg["budget"]["raw_frames"],
            "audit_seed_base": cfg["evaluation"]["audit_seed_base"],
            "audit_episodes": cfg["evaluation"]["audit_episodes"],
            "per_episode_raw_return": audit_returns,
            "mean_raw_return": float(np.mean(audit_returns)),
            "std_raw_return": float(np.std(audit_returns)),
        }
        summary = run_dir / "summary"
        summary.mkdir(exist_ok=True)
        _write_json(summary / "scores.json", scores)
        _atomic_write(summary / "scores.csv",
                      "episode,raw_return\n" +
                      "".join(f"{i},{r:.4f}\n" for i, r in enumerate(audit_returns)))
        _atomic_write(summary / "scores.md",
                      f"# {scores['game']} ({scores['variant']})\n\n"
                      f"- steps: {steps_total:,} ({scores['raw_frames']:,} raw frames)\n"
                      f"- audit mean raw return: **{scores['mean_raw_return']:.1f}**"
                      f" ± {scores['std_raw_return']:.1f} "
                      f"({len(audit_returns)} episodes)\n")

        wall = time.perf_counter() - t_start
        prov = _build_provenance(cfg, fingerprint, seed, wall, eval_seconds,
                                 ckpt_seconds, steps_done)
        _write_json(run_dir / "provenance.json", prov)

        # completion gate: verify artifact hashes before flipping state
        ok = all([
            (run_dir / "summary" / "scores.json").exists(),
            (run_dir / "provenance.json").exists(),
            _verify_generation(run_dir / "checkpoints" / "final", fingerprint),
        ])
        if not ok:
            _write_json(run_dir / "run_state.json",
                        {"state": "FAILED", "reason": "artifact verification failed",
                         "fingerprint": fingerprint})
            return 1
        _write_json(run_dir / "run_state.json",
                    {"state": "COMPLETED", "finished_at": _now_iso(),
                     "fingerprint": fingerprint,
                     "mean_raw_return": scores["mean_raw_return"]})
        print(f"[run] COMPLETED steps={steps_done} "
              f"mean_raw={scores['mean_raw_return']:.1f} wall={wall:.0f}s")
        return 0
    except KeyboardInterrupt:
        _write_json(run_dir / "run_state.json",
                    {"state": "INTERRUPTED", "fingerprint": fingerprint})
        raise
    except Exception as e:  # noqa: BLE001
        _write_json(run_dir / "run_state.json",
                    {"state": "FAILED", "reason": repr(e),
                     "fingerprint": fingerprint})
        raise
    finally:
        train_log.close()
        tb.close()
        env.close()


def _fake_eval_env(cfg, eval_seed):
    import gymnasium as gym
    return gym.make(cfg["game"]["env_id"])


def _run_periodic(agent, cfg, eval_hook):
    ev = cfg["evaluation"]
    returns = []
    for i in range(ev["periodic_episodes"]):
        env = eval_hook(cfg, ev["periodic_seed_base"] + i)
        r, _ = eval_lib.run_episode(agent, env)
        returns.append(r)
        env.close()
    return float(np.mean(returns)), returns


def _run_audit(agent, cfg, eval_hook):
    ev = cfg["evaluation"]
    returns = []
    for i in range(ev["audit_episodes"]):
        env = eval_hook(cfg, ev["audit_seed_base"] + i)
        r, _ = eval_lib.run_episode(agent, env)
        returns.append(r)
        env.close()
    return returns


def _build_provenance(cfg, fingerprint, seed, wall, eval_s, ckpt_s, steps):
    prov = {
        "command": f"experiment.py run (variant={cfg['experiment']['variant']})",
        "config_fingerprint": fingerprint,
        "seed": seed,
        "started_env": cfg["game"]["env_id"],
        "steps": steps,
        "raw_frames": cfg["budget"]["raw_frames"],
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "none"),
        "os": platform.platform(),
        "start_end_wall_seconds": round(wall, 1),
        "eval_seconds": round(eval_s, 1),
        "checkpoint_seconds": round(ckpt_s, 1),
        "training_fps": round(steps / max(wall - eval_s - ckpt_s, 1e-9), 1),
        "source_sha256": {
            p.name: _sha256_file(p) for p in sorted(MRQ_DIR.glob("*.py"))
        },
        "env_source_sha256": {
            p.name: _sha256_file(p) for p in sorted(
                (MRQ_DIR.parents[1] / "baselines_common" / "envs").glob("*.py"))
        },
        "recorded_at": _now_iso(),
    }
    if cfg["provenance"].get("record_pip_freeze"):
        try:
            import subprocess
            out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                 capture_output=True, text=True, timeout=120)
            # drop editable installs: local checkout paths are machine
            # identity, not environment identity
            prov["pip_freeze"] = [
                line for line in out.stdout.strip().splitlines()
                if not line.startswith("-e ")
                and not line.startswith("# Editable")
            ]
        except Exception:  # noqa: BLE001
            prov["pip_freeze"] = "unavailable"
    return prov


# ------------------------------------------------------------------- campaign

def _cmd_campaign(args) -> int:
    cfg = load_config(args.config)
    campaign_cfg_path = Path(args.config)
    # A per-game file passed to campaign = single-game campaign (useful for
    # tests and reruns); a suite manifest runs its full fixed game list.
    if cfg.get("game", {}).get("slug"):
        games = [cfg["game"]["slug"]]
        games_dir = campaign_cfg_path.parent
        if (games_dir / "games").exists():
            games_dir = games_dir / "games"
    else:
        games = cfg["suite"]["games"]
        games_dir = _resolve_games_dir(
            campaign_cfg_path, cfg["experiment"]["campaign"])
    run_root = Path(args.run_root or
                    (Path(cfg["output"]["run_root"]) /
                     cfg["experiment"].get("id", "campaign")))
    run_root.mkdir(parents=True, exist_ok=True)
    report = {"campaign": cfg["experiment"]["id"], "runs": [],
              "completed": 0, "failed": 0, "skipped": 0}
    for slug in games:
        game_cfg = games_dir / f"{slug}.yaml"
        if not game_cfg.exists():
            report["failed"] += 1
            report["runs"].append({"slug": slug, "status": "missing-config"})
            continue
        run_dir = run_root / slug
        state_f = run_dir / "run_state.json"
        if state_f.exists():
            state = json.loads(state_f.read_text())
            if state.get("state") == "COMPLETED":
                rc = _apply_smoke(load_config(str(game_cfg))) if args.smoke \
                    else load_config(str(game_cfg))
                fp = config_fingerprint(rc)
                if _verify_completed_run(run_dir, fp):
                    report["skipped"] += 1
                    report["runs"].append({"slug": slug, "status": "skipped"})
                    continue
        rc = _run_one(str(game_cfg), run_dir, smoke=args.smoke, resume=False)
        report["completed" if rc == 0 else "failed"] += 1
        report["runs"].append({"slug": slug,
                               "status": "completed" if rc == 0 else "failed"})
        _write_json(run_root / "campaign_report.json", report)
    _write_json(run_root / "campaign_report.json", report)
    print(f"[campaign] completed={report['completed']} "
          f"skipped={report['skipped']} failed={report['failed']}")
    return 0 if report["failed"] == 0 else 1


# -------------------------------------------------------------------- status

def _cmd_status(args) -> int:
    run_dir = Path(args.run_dir)
    state = json.loads((run_dir / "run_state.json").read_text())
    print(json.dumps(state, indent=2))
    return 0


def _cmd_summarize(args, capture: bool = False):
    run_dir = Path(args.run_dir)
    scores = json.loads((run_dir / "summary" / "scores.json").read_text())
    line = (f"{scores['game']} ({scores['variant']}): audit mean raw "
            f"{scores['mean_raw_return']:.1f} ± {scores['std_raw_return']:.1f} "
            f"over {scores['audit_episodes']} episodes "
            f"@ {scores['steps_requested']:,} steps")
    if capture:
        return 0, line
    print(line)
    return 0


# ------------------------------------------------------------------- archive

def _cmd_archive(args) -> int:
    run_dir = Path(args.run_dir)
    state = json.loads((run_dir / "run_state.json").read_text())
    if not _verify_completed_run(run_dir, state.get("fingerprint")):
        print("archive refused: run is not a verifiable COMPLETED run "
              "(state, fingerprint, summary, provenance, and the final "
              "checkpoint generation must all validate)", file=sys.stderr)
        return 1
    archive_root = Path(args.archive_root)
    dest = archive_root / f"{run_dir.name}_{state['finished_at'].replace(':', '')}"
    if dest.exists():
        print("archive refused: destination exists (immutable)",
              file=sys.stderr)
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=str(archive_root)))
    shutil.copytree(run_dir, tmp / run_dir.name)
    manifest = {"source": str(run_dir), "archived_at": _now_iso(),
                "files": _hash_tree(tmp / run_dir.name)}
    _write_json(tmp / run_dir.name / "archive_manifest.json", manifest)
    os.replace(tmp / run_dir.name, dest)
    os.rmdir(tmp)
    print(f"[archive] {dest}")
    return 0


# ----------------------------------------------------------------------- CLI

def cli(argv=None, capture: bool = False) -> int:
    parser = argparse.ArgumentParser("mrq-experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate")
    p.add_argument("--config", required=True)
    p.set_defaults(func=lambda a: 0 if _validate(a) else 1)

    p = sub.add_parser("preflight")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_cmd_preflight)

    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("campaign")
    p.add_argument("--config", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run-root", default=None)
    p.set_defaults(func=_cmd_campaign)

    p = sub.add_parser("status")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("summarize")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=_cmd_summarize)

    p = sub.add_parser("archive")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--archive-root", required=True)
    p.set_defaults(func=_cmd_archive)

    args = parser.parse_args(argv)
    func = args.func
    if capture:
        return func(args, capture=True)
    return func(args)


def _validate(args) -> bool:
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"validate FAILED: {e}", file=sys.stderr)
        return False
    print(f"validate OK fingerprint={config_fingerprint(cfg)} "
          f"game={cfg.get('game', {}).get('slug', '<suite>')} "
          f"steps={cfg['budget']['steps_requested']:,}")
    return True


def _cmd_run(args) -> int:
    if args.seed is not None:
        # explicit seed override must be visible in the resolved config
        raise SystemExit("--seed override not supported: seed lives in the "
                         "config (frozen identity); edit the YAML instead")
    run_dir = Path(args.run_dir) if args.run_dir else (
        Path(load_config(args.config)["output"]["run_root"]) /
        load_config(args.config)["experiment"]["id"] /
        load_config(args.config)["game"]["slug"])
    return _run_one(args.config, run_dir, smoke=args.smoke, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(cli())
