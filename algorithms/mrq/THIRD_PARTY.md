# Third-party source

**MRQ** — "Towards General-Purpose Model-Free Reinforcement Learning",
Fujimoto, D'Oro, Zhang, Tian, Rabbat (Meta FAIR),
[arXiv:2501.16142](https://arxiv.org/abs/2501.16142).

- Upstream repository: https://github.com/facebookresearch/MRQ
- Pinned commit: `073adcd6b30f84694c241e00bd684db50242a215`
  (upstream HEAD at the snapshot date)
- Snapshot date: 2026-09-14
- Upstream license: **CC BY-NC 4.0** (non-commercial). The upstream license
  text is NOT distributed in this repository because no upstream file is
  included; it is available in the upstream repository linked above.

This directory contains a **clean-room reimplementation** written from the
paper, with the upstream implementation used for behavioral verification
only. **No upstream file is imported, copied, or vendored** here, and no
upstream file will ever be added to this repository. The read-only reference
snapshot used during development is retained outside this repository as
private provenance.

Because the upstream license is CC BY-NC 4.0, the clean-room rule applied here
is stricter than for MIT-licensed upstreams (e.g. the PPO-PyTorch line): the
source code in this directory is written fresh — variable names, structure,
and comments are our own — and its semantics were cross-checked against the
paper and the reference's observable behavior only. The code in this
directory is therefore original work under the repository's Apache-2.0
license (see `NOTICE.md`).

When citing MRQ, cite the arXiv paper and the upstream repository URL, and
state that this implementation is an independent reimplementation.
