# Atari-16 Game Selection

## Selection policy

This suite keeps all six Atari games on which MuZero did not exceed the human baseline: Montezuma's Revenge, Pitfall, Private Eye, Skiing, Solaris, and Venture. Ten additional games provide broader skill coverage. Within each skill family, lower PPO human-normalized performance was preferred; a focused game was also retained when relying only on multi-skill games could hide a missing capability.

Together, the 16 games cover position coverage and collection, dodging, jumping and climbing, blocking and interception, vehicle control, racing, melee combat, ranged combat, planning and puzzle solving, navigation and exploration, defense and rescue, precise timing, and resource management.

## Scores and rationale

| Game | Selection rationale | Agent57 raw score | Original PPO raw score | PPOSB3 raw score |
| --- | --- | ---: | ---: | ---: |
| Alien | Low PPO HNS; focused maze navigation, collection, and dodging representative. | 297638.17 | 1850.3 | 1001.33 |
| Asteroids | Very low PPO HNS; adds free-rotation vehicle control, ranged combat, and dodging. | 150854.61 | 2097.5 | 1377.00 |
| Bowling | The suite's most focused precision-and-timing task and a low PPO-HNS game. | 251.18 | 40.1 | NaN |
| Chopper Command | Low PPO HNS; adds vehicle control, ranged combat, and active defense. | 999900.00 | 3516.3 | NaN |
| Enduro | Lowest-PPO-HNS reported driving race; complements Skiing with vehicle-based racing. | 2367.71 | 758.3 | NaN |
| Frostbite | Very low PPO HNS; adds reactive jumping, collection, dodging, and timing. | 541280.88 | 314.2 | NaN |
| Gopher | Lowest practical cross-skill choice combining interception, melee action, and defense. | 117777.08 | 2932.9 | NaN |
| Kung Fu Master | Lowest-PPO-HNS melee game; isolates close combat better than the mandatory set. | 206845.82 | 23310.3 | NaN |
| Montezuma's Revenge | Mandatory: MuZero did not exceed human; sparse rewards, exploration, jumping, and long-horizon planning. | 9352.01 | 42.0 | NaN |
| Pitfall | Mandatory: MuZero did not exceed human; exploration, platform traversal, collection, and hazard avoidance. | 18756.01 | -32.9 | NaN |
| Private Eye | Mandatory: MuZero did not exceed human; driving, evidence collection, planning, and time management. | 79716.46 | 69.5 | NaN |
| Seaquest | Lowest PPO HNS among defense/rescue games; also tests oxygen and resource management. | 999997.63 | 1204.5 | NaN |
| Skiing | Mandatory: MuZero did not exceed human; focused racing, trajectory control, and obstacle avoidance. | -4202.60 | NaN | NaN |
| Solaris | Mandatory: MuZero did not exceed human; navigation-heavy vehicle combat and resource management. | 44199.93 | NaN | NaN |
| Tennis | A relatively low-PPO-HNS, focused blocking, interception, and return-timing task. | 23.84 | -14.8 | NaN |
| Venture | Mandatory: MuZero did not exceed human; maze navigation, ranged combat, collection, and dodging. | 2623.71 | 0.0 | NaN |

## Score provenance

- Agent57 scores are the reported Agent57 means in Badia et al., *Agent57: Outperforming the Atari Human Benchmark*, Appendix H.4.
- Original PPO scores are the mean final scores in Schulman et al., *Proximal Policy Optimization Algorithms*, Table 6. `NaN` means the paper did not report that game.
- PPOSB3 scores are deterministic 30-episode final raw-return means from this repository's 10M-SB3-timestep, seed-0 protocol. Alien and Asteroids were completed in the earlier Atari-57 campaign and imported without retraining after semantic and checkpoint-hash verification. `NaN` means no completed PPOSB3 run was available when this document was created.
- Scores across the three systems are reported under different evaluation protocols and should not be treated as a strict same-protocol ranking.
