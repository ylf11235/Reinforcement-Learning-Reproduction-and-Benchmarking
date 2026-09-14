"""Generate byte-stable checked-in formal EADream Atari-16 YAML inputs."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


GAMES = {
    "alien": "ALE/Alien-v5", "asteroids": "ALE/Asteroids-v5",
    "bowling": "ALE/Bowling-v5", "chopper_command": "ALE/ChopperCommand-v5",
    "enduro": "ALE/Enduro-v5", "frostbite": "ALE/Frostbite-v5",
    "gopher": "ALE/Gopher-v5", "kung_fu_master": "ALE/KungFuMaster-v5",
    "montezuma_revenge": "ALE/MontezumaRevenge-v5", "pitfall": "ALE/Pitfall-v5",
    "private_eye": "ALE/PrivateEye-v5", "seaquest": "ALE/Seaquest-v5",
    "skiing": "ALE/Skiing-v5", "solaris": "ALE/Solaris-v5",
    "tennis": "ALE/Tennis-v5", "venture": "ALE/Venture-v5",
}
FIRE_GAMES = {"bowling", "tennis"}
ROM_SHA256 = {
    "alien": "9cd556c7d7f55aeb78fc5add45d74ce889c6e22cba6bb535ef1e8bd6b6c764c1",
    "asteroids": "5ba6f91851b2331a37a3fe6a950c9b464fd764cb035a3d25d4b4388e9b26768f",
    "bowling": "dff44a85289f5e9f760254e4806d09038326cdfc42eba3e1cd7c64ed6e0b4dce",
    "chopper_command": "055637282252b5378a2b7df573726e85afa91264ffb7243d23fab9518b3c207d",
    "enduro": "6045c8be78c7d0bec29040022543a8c0b9e3672b50005a94bf0166f0f73be3d9",
    "frostbite": "cbfdad89480def69d922adf16a1d89d55a6cb515929edf74e5bea884c9fb7834",
    "gopher": "b4aff03aeb0fb1c8f4914b5a329f51430f00a2d9350aba24573ad78032dd4697",
    "kung_fu_master": "3f6501a649ad83e970a25827bd492c56128c36535ae2c96f94bab39b27f939ac",
    "montezuma_revenge": "69d363a747549817599cace65572700dab9ea33f3a5f91979454fdfbeaa837c0",
    "pitfall": "c56c99d9e00136a015a485851e0e925c6327ea2b20aa7d3daecfbd7f9afcfdf0",
    "private_eye": "c8f9ce1b2e804b6778dbc20da0c935b2af7d7e998ad372d44df66f16eb4b98ce",
    "seaquest": "fbc29f4678f69ac27fe46da298c8c22f98cba2a3e5491e6b308fd9bf68d2ee43",
    "skiing": "e8a96a74c05493c5394a097deb7dc339a0cf37fe373d93e4c19f9e6884eff36c",
    "solaris": "0afa36e5f4d8d77e38716673c96cda3ad11d421283f5a5ae77ec7a01e7718a4e",
    "tennis": "b829e42751fcbc93ad7be8c6bd41e00083aad74d1f9f869f40898d21fb34133b",
    "venture": "5cdd323b8f817cc5d953910db97c20d9b7203316a8996663fdc953f2e29ee80e",
}

FORMAL = {
    "schema_version": 1,
    "experiment": {"id": "eadream_atari16_100k_v1", "algorithm": "eadream", "formal": True, "result_kind": "atari16_100k"},
    "suite": {"id": "atari16", "manifest": "eadream/configs/atari16.yaml", "seeds": [0, 1, 2, 3, 4]},
    "budget": {"unit": "agent_interactions", "agent_interactions": 100_000, "ale_frames_nominal": 400_000},
    "runtime": {"device": "cuda:0", "require_cuda": True, "precision": "float32", "compile": False, "num_envs": 1, "python_version": "3.10.20", "torch_version": "2.11.0+cu130", "torch_cuda_version": "13.0", "deterministic_algorithms": False},
    "environment": {"frameskip": 1, "action_repeat": 4, "screen_size": 64, "grayscale": False, "frame_stack": 1, "noop_max_exclusive": 30, "repeat_action_probability": 0.0, "terminal_on_life_loss": False, "minimal_action_set": True, "max_num_frames_per_episode": 108_000, "max_agent_steps_per_episode": 27_000, "reward_clip": False, "max_pool_semantics": "source_fixed_two_buffer"},
    "events": {"history": 500, "var_threshold": 16.0, "detect_shadows": True, "learning_rate": -1.0, "closing_kernel": 3, "closing_shape": "ellipse", "event_pred_ratio": 0.05},
    "model": {"dyn_stoch": 64, "dyn_discrete": 32, "dyn_deter": 512, "dyn_hidden": 512, "dyn_rec_depth": 1, "units": 512, "activation": "SiLU", "normalization": "LayerNorm", "layer_norm_eps": 1e-3, "unimix_ratio": 0.01, "initial": "learned", "cnn_depth": 32, "cnn_kernel": 4, "cnn_min_resolution": 4, "encoder_embed_dim": 4096, "image_input_scale": "divide_255", "encoder_input_offset": -0.5, "reconstruction_target_range": "zero_one", "encoder_attention": True, "encoder_attention_kernel": 3, "encoder_attention_ratio": 2, "decoder_attention": False, "event_decoder_attention": False, "predict_from_prior": True, "first_frame_prediction": False, "previous_current_event_input": False, "mae_ratio": 1.0, "multihead_rssm_input": False, "dense_event_filter": True, "harmony": True, "kl_free": 1.0, "dyn_scale": 0.5, "rep_scale": 0.1, "event_loss_scale": 0.5, "event_focal_alpha": 0.15, "event_focal_gamma": 4.0, "image_attention_weight": 0.5, "head_layers": 2, "actor_distribution": "onehot", "scalar_distribution": "symlog_disc", "scalar_buckets": 255, "scalar_low": -20.0, "scalar_high": 20.0, "decoder_output_offset": 0.5, "event_output_sigmoid": True, "reward_outscale": 0.0, "continuation_outscale": 1.0, "actor_outscale": 1.0, "value_outscale": 0.0},
    "training": {"prefill": 2500, "pretrain_batches": 100, "batch_size": 16, "batch_length": 64, "train_ratio": 1024, "imag_horizon": 15, "discount": 0.997, "lambda": 0.95, "dataset_size": 1_000_000, "sequence_sampler_seed": 0, "reward_ema": True, "reward_ema_alpha": 0.01, "imag_gradient": "reinforce", "imag_gradient_mix": 0.0, "exploration_behavior": "greedy", "exploration_until": 0, "detect_anomaly_world_model": True, "actor_entropy": 3e-4, "slow_value_update": 1, "slow_value_fraction": 0.02, "optimizer": "adamw", "optimizer_effective_eps": 1e-8, "optimizer_effective_betas": [0.9, 0.999], "optimizer_effective_weight_decay": 0.01, "model_lr": 1e-4, "model_grad_clip": 1000.0, "actor_lr": 3e-5, "actor_grad_clip": 100.0, "value_lr": 3e-5, "value_grad_clip": 100.0},
    "evaluation": {"deterministic": True, "formal_episodes": 100, "formal_seed_source": "train_seed_fresh_env", "eval_state_mean": False, "rssm_sampling": True, "evaluator_rng": "source_seed_before_model_construction", "audit_seeds": list(range(20000, 20030)), "no_fire_diagnostic_games": ["bowling", "tennis"], "no_fire_diagnostic_steps": 200, "periodic_anchor_agent_interactions": 2_500, "periodic_every_agent_interactions": 7_500, "periodic_nominal_ale_frames": 30_000},
    "checkpoint": {"save_every_agent_interactions": 7_500, "first_periodic_agent_interactions": 10_000, "keep_periodic": 2, "validate_after_write": True, "immutable_milestones": [100_000]},
    "video": {"enabled": True, "checkpoint": "best", "world_model_prediction": True, "episode_seeds": [30000, 30001, 30002], "source": "native_ale_rgb"},
    "output": {"campaign_root": "eadream/runs/atari16/eadream_atari16_100k_v1", "archive_root": "eadream/archives/atari16/eadream_atari16_100k_v1", "archive_scope": "result"},
    "campaign": {"scheduling": "serial", "on_failure": "record_and_continue", "automatic_resume": True, "skip_completed": True, "max_retries_per_run": 1},
    "provenance": {"reference_commit": "269f71af5b3510fbcdb2c7d3ebeb22b1a15f5241", "reference_manifest": "eadream/reference_manifest.json", "declared_actor_value_eps": 1e-5, "declared_extra_weight_decay": 0.0},
}


def game_document(slug: str, env_id: str) -> dict:
    document = copy.deepcopy(FORMAL)
    document["game"] = {"slug": slug, "env_id": env_id, "rom_sha256": ROM_SHA256[slug]}
    document["environment"]["start_protocol"] = "fire" if slug in FIRE_GAMES else "none"
    return document


def _write_yaml(path: Path, document: dict) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8", newline="\n")


def main(config_dir: Path | None = None) -> None:
    config_dir = Path(__file__).resolve().parent if config_dir is None else config_dir
    games_dir = config_dir / "atari16" / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "suite": {"id": "atari16", "expected_game_count": 16}, "games": [{"slug": slug, "env_id": env_id, "rom_sha256": ROM_SHA256[slug]} for slug, env_id in GAMES.items()]}
    _write_yaml(config_dir / "atari16.yaml", manifest)
    for slug, env_id in GAMES.items():
        _write_yaml(games_dir / f"{slug}.yaml", game_document(slug, env_id))
    campaign = {
        "schema_version": 1,
        "campaign_id": "eadream_atari16_100k_v1",
        "manifest": "eadream/configs/atari16.yaml",
        "game_directory": "eadream/configs/atari16/games",
        "games": list(GAMES),
        "seeds": [0, 1, 2, 3, 4],
        "campaign": copy.deepcopy(FORMAL["campaign"]),
        "output": {"campaign_root": FORMAL["output"]["campaign_root"], "archive_root": FORMAL["output"]["archive_root"]},
    }
    _write_yaml(config_dir / "atari16_campaign.yaml", campaign)
    _write_yaml(config_dir / "alien_pilot_100k.yaml", game_document("alien", GAMES["alien"]))


if __name__ == "__main__":
    main()
