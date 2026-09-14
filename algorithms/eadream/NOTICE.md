# EADream Native Reproduction Notice

This native implementation reproduces behavior adapted from the GPL-3.0
EADream reference implementation (upstream repository: MarquisDarwin/EAWM) at
commit `269f71af5b3510fbcdb2c7d3ebeb22b1a15f5241`.
The tracked raw-file provenance is recorded in `reference_manifest.json`.

Native deviations are deliberate and recorded for experiment provenance:

- Gymnasium/ALE-Py replaces the reference Gym/atari-py stack, with explicit ALE seeding.
- Bowling and Tennis use the formal FIRE start protocol.
- Completed replay episodes are durably persisted.
- Checkpoints are complete and support statistical (not bitwise) resume.
- Evaluation and logging use isolated RNG ownership.
- Run lifecycle, campaign management, provenance, and archive management are native additions.

This notice is not a legal determination. Distribution of adapted material must
retain the upstream GPL-3.0 obligations and receive a separate license audit.
