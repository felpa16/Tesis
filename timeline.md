# Project Timeline

An ongoing journal of concrete runs, bugs, measurements and decisions.

`CLAUDE.md` describes what the project *is* — the architecture and the research
design. This file records *how it actually went*: what was run, what broke, what
the numbers came out to, and why each choice was made. When a decision here
changes the project's design rather than just its execution, `CLAUDE.md` is
updated too and the entry says so.

Newest entries at the bottom.

---

## 2026-08-18 — Data acquisition from SHS100K

SHS100K ships as YouTube IDs, not audio, so the dataset has to be scraped.
`scripts/download_shs100k.py` drives yt-dlp over the cover list.

**Findings**

* ~3.84 MB per track at the chosen encoding; ~320 GB for the full dataset.
* **76% yield.** The remaining ~24% is link rot — videos deleted, region-locked,
  or made private since SHS100K was published.
* The bottleneck is **YouTube rate limiting, not bandwidth**: ~765 tracks/hour.
  A full scrape is therefore a ~6-day job, not an afternoon.

**Decisions**

* **Keep the inter-download sleep.** It was removed once as an optimisation
  (`ea0518a`) and YouTube began blocking outright; it was restored in `ed112ff`.
  The sleep is load-bearing, not a politeness gesture.
* **yt-dlp authenticates with Chrome cookies** — a meaningful fraction of tracks
  are otherwise unavailable.
* **Stop tracking generated data in git** (`9df7b1a`). `data/` holds hundreds of
  GB of audio and derived `.npz`; it does not belong in the repo.
* **All dataset storage lives on S3**, not on the laptop. The local machine has
  8 GB RAM and a tight disk and cannot hold the corpus.

---

## 2026-08-19 — Audio health check before upload

`scripts/check_audio.py` verifies each downloaded file is decodable before it is
pushed to S3, so that a corrupt download fails here rather than deep inside a
training run.

**Bug** — the first version flagged *every* file as unreadable (fixed in
`b449977`). Worth remembering as a class: a validator that rejects 100% of its
input is far more likely to be broken than the input is.

---

## ~2026-08-25 → 2026-08-27 — Phase-1 preprocessing (box A, CPU instance)

Cover alignment for the Phase-1 subset: **7,469 tracks**. Two stages of
`scripts/align_covers.py` — beat-synchronous `chroma_cqt`, then OTI over 12
circular shifts followed by numba Smith-Waterman — producing one
`data/alignments/{split}/{song}_{verA}_{verB}.npz` per pair.

**Decisions**

* **Alignment score threshold 0.2.** Of ~20,459 candidate pairs, **8,570 survive
  (42%)**. The discarded 58% are the intended casualties: mislabeled pairs,
  remixes, medleys, and covers too structurally different to align.
* **`--workers 2` for the chroma stage** on the 8 GB local machine; higher
  worker counts swap.
* Box A was a CPU-only instance — chroma and Smith-Waterman are CPU work and
  renting a GPU for them is waste. It was **terminated** once the alignments
  were uploaded to S3.

---

## ~2026-08-27 → 2026-08-28 — Phase-1 training environment (box B, g5.2xlarge)

Six consecutive blockers between "instance launched" and "training running".
Recorded in full because most are environmental and will recur on the next box.

### 1. `No module named 'numpy'` in a bare shell, `No module named 'zuko'` in tmux

Two different interpreters. The Deep Learning AMI does **not** put PyTorch in the
system Python — it lives in a venv at `/opt/pytorch` (older AMIs: a conda env
named `pytorch`). Separately, `tmux new -s foo` attaches to an *existing* tmux
server, and panes inherit the environment that server was born with, so a tmux
session started before the venv existed never sees it.

**Fix** — `echo 'source /opt/pytorch/bin/activate' >> ~/.bashrc`, then
`tmux kill-server` to force a fresh server. `DATA` and `HF_HOME` were persisted
the same way.

### 2. `FileNotFoundError: 'manifests/train/tracks.jsonl'` — note the missing prefix

`$DATA` was unset in that shell, so `--data-root ""` became `Path("")` → `.`.
Manifest paths are stored **relative to the data root**, so the correct fix is to
export `DATA`, *not* to `cd data` — the root is the directory containing
`audio/`, `alignments/` and `manifests/`.

**Open item:** `scripts/train.py` should reject an empty `--data-root` rather
than silently resolving it to the current directory.

### 3. `FileNotFoundError: .../alignments/train/897_60711_845439.npz` inside a worker

Not a naming bug. The runbook extracted the alignment tarball with
`aws s3 cp ... - | tar xzf -`; without `pipefail` the pipeline's exit status is
tar's, so a truncated download reported success.

**Fix** — download to a file, verify with `tar tzf … | grep -c '\.npz'`, then
extract.

### 4. `no usable aligned pairs in split 'train'`

Correct behaviour, wrong input. `scripts/build_manifest.py:109` globs
`alignment_dir(...).glob("*.npz")`, so it can only reference alignments that
exist **on the machine it runs on**. Box A was already terminated; the directory
simply wasn't there yet.

**Invariant established: build the manifest on the box you train on**, after the
alignments have landed.

### 5. `torch.OutOfMemoryError` at 21.42 GiB on a 22 GiB A10G

Default batch was 8 pairs + 8 candidates + 8 tracks = 24 windows of
(1500, 1024).

**Fix** — `--batch-pairs 4 --batch-tracks 4` (12 windows) plus
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Measured: ~3.3 GB fixed,
~1.28 GB marginal per window.

**Correction worth keeping:** the first remedy proposed was `--mert-micro-batch 2`.
That does **not** reduce peak memory. `LayerMix.forward` uses
`torch.einsum("blnd,l->bnd", ...)`, and walking the autograd graph shows the
resulting `BmmBackward0` saves each chunk's hidden states for backward — so
micro-batching MERT changes nothing about the peak. The docstring in
`extract_mixes` still claims the full `(B, 25, N, 1024)` tensor never
materialises; that is true forward-only and misleading in the presence of
autograd.

### 6. `Permissions 0644 … are too open` on the SSH key

macOS/OpenSSH refuses a world-readable private key. `chmod 400` on the `.pem`.
TensorBoard is reached over an SSH port-forward (`-L 6006:localhost:6006`) rather
than by opening 6006 in the security group — TensorBoard has no authentication.

---

## 2026-08-28 — Phase-1 run #1 completes

**Configuration**

| | |
|---|---|
| instance | g5.2xlarge (A10G, 22 GiB) |
| subset | 7,469 tracks / 8,570 aligned pairs |
| batch | `--batch-pairs 4 --batch-tracks 4` → 12 windows/step |
| window | 20 s (1,500 MERT frames) |
| steps | 2,142/epoch × 10 epochs = **21,420** |
| speed | ~1.31 s/step; 2,806 s/epoch |
| total | ~7.8 h, ≈ $10 |
| peak VRAM | 18,735 / 23,028 MB |

**Terminal losses** (mean over the last 24 logged steps)

```
recon 1.99   swap 2.01   contrastive 0.36   cycle 0.0025   decorrelation 0.068
```

Note the printed values are **raw**; `total` applies the config weights
(recon 1.0, contrastive 1.0, swap 0.25, cycle 0.25, decorrelation 0.1).

**Layer-weight stationarity** — cosine to the previous epoch reached
`0.999999` on both branches. Under the freeze criterion as originally written,
Phase 1 was complete.

---

## 2026-08-28 — Phase-1 audit: the freeze criterion was insufficient

Before committing to Phase 2 (~7 TB of cached mixes), `scripts/inspect_phase1.py`
was written to ask what the weights converged *to*, and whether `recon` is
beating the trivial predictors.

### Layer mix

| | content | style |
|---|---|---|
| entropy (uniform = 3.2189) | 3.2055 | **3.2188** |
| max weight (uniform = 0.0400) | 0.0502 | **0.0414** |
| cos to uniform | 0.9868 | **0.999884** |
| ‖deviation from uniform‖ | 0.0328 | **0.0030** |
| peak layers | 7–10 | 2–5 |

`cos(content, style) = 0.9876`.

**Content worked.** A clean unimodal hump over layers 6–11 — the middle of the
stack, where MERT probing places pitch, harmony and chord information — and it is
backed by a contrastive loss well below chance (0.36 vs. `ln(4) = 1.386` at
`batch_pairs=4`).

**Style did not.** Its entropy matches uniform to four decimals; it is a flat
average over all 25 layers. Its *shape* is not random — it tilts smoothly toward
layers 2–5, where timbre and acoustic detail live, so it was pushing in the right
direction — but at ~1/11 the amplitude of content's. As a learned selection it is
vacuous.

### Reconstruction vs. trivial predictors

```
predict dataset mean      1.9526     <- learning nothing
MODEL recon               1.9348
predict per-window mean   1.8499     <- one 1024-d vector per window
```

The **entire gap between the two baselines is 5.3% of the variance.** The other
94.7% is frame-to-frame variation *inside* a window: MERT at 75 fps is mostly
high-frequency detail that no 16×256 latent set can represent. The model captured
0.9% of total variance — 17.3% of the available between-window range, less than a
single per-window mean vector would achieve.

Decoder output std is **0.0847** against a target std of 1.0. That is the
MSE-optimal shrinkage response to an unpredictable target: a predictor with
correlation ρ outputs ρ·z, and the per-branch MSE implies ρ ≈ 0.18.

Two innocent explanations were checked and ruled out:

* *Standardisation amplifying near-constant dimensions into noise* — the
  standardiser is healthy: `var` min 6.17, median 12.95. No dead dimensions.
* *The 2.0 yardstick being wrong* — the measured dataset-mean baseline is 1.9526,
  close to the predicted 2.0 (`recon` sums two unit-variance MSEs).

### Diagnosis

`recon` is ~95% irreducible noise by construction. The learnable signal is a 5%
sliver competing against the gradient variance of the other 95%. This is why the
**style** layer weights never moved: `recon` and `swap` are their only gradient
source. Content escaped because the contrastive loss feeds it independently.

`cycle = 0.0025` says the decode→re-encode path is far better optimised than the
decode-to-match-reality path — a mild version of the encoder-decoder collusion
that `CLAUDE.md` anticipates — but shrinkage under uncertainty explains most of
the observed behaviour without invoking steganography.

### Decisions

1. **Do not start Phase 2 on these weights.** Phase 1 costs ~8 h and ~$10;
   re-caching 7 TB does not. Caching a flat style mix chosen by a dead objective
   would bake the problem into an expensive artifact, and the two streams would be
   near-redundant.
2. **The content weights are sound**; the style weights are not. The failure is
   one-sided and the audit says which side.
3. **Change the reconstruction target to a temporally pooled one** — average the
   mix over ~16-frame blocks (≈94 target frames per 20 s window) so the loss stops
   being dominated by unlearnable detail. Exposed as a `recon_pool` config field,
   ablatable by setting it to 1. **→ `CLAUDE.md` updated** (Decoder,
   Reconstruction objective 2a, Open Research Questions).
4. **Turn on the cosine term** (`cosine_weight` is currently 0.0) in the same
   ablation — it is scale-free and less dominated by high-frequency amplitude.
5. **Do not grow the decoder.** The shrinkage result says capacity is not the
   binding constraint, and `CLAUDE.md` already warns that a stronger decoder
   weakens disentanglement pressure.
6. **Amend the Phase-1 freeze criterion.** Per-epoch cosine ≈ 1 is necessary but
   not sufficient — it is also exactly what a parameter that never moved produces.
   **→ `CLAUDE.md` updated** (Feature caching strategy). The criterion now also
   requires a meaningful departure from uniform and evidence that the branch's
   driving objective is learning.

**Success criterion for run #2:** style's cos-to-uniform falls to roughly
content's current 0.987, and `recon` lands clearly below the per-window-mean
floor rather than above it.

---

## 2026-08-28 — Pooled reconstruction target implemented

`recon_pool` (default **16**) added to `LossConfig`, exposed as `--recon-pool`.
A 20 s window's 1,500-frame target becomes 93 pooled frames; `recon_pool = 1`
restores the old behaviour and is the ablation.

**Design choices made during implementation**

* **Pool the target, not the encoder input.** The encoders still consume the
  mixes at full 75 Hz resolution, exactly as `CLAUDE.md` specifies. Only the
  decoder's target changes.
* **Pool *before* standardizing.** Averaging shrinks variance, so pooling
  already-standardized values would drop the target below unit variance and
  break the "predict the dataset mean scores 1.0 per branch" yardstick. The
  Standardizer is now updated on the pooled target so its statistics describe
  what the loss is actually scored against. Verified numerically: the
  dataset-mean baseline stays at 1.0000 (pool 1) / 0.9999 (pool 16).
* **The decoder emits at pooled resolution** rather than emitting 1,500 frames
  that are then pooled before the MSE. Pooling is linear so the two are
  equivalent in what they score — but emitting 1,500 frames would leave the
  high-frequency component of the decoder's output completely unconstrained,
  which is exactly the free channel that `cycle` could hide latents in. Decoding
  at 93 frames removes that channel and is ~16× cheaper.
* **`cycle` runs at pooled resolution too**, for the same reason. This means the
  re-encode path sees 93-frame sequences while the encoder normally sees 1,500 —
  a domain shift, but the alternative reintroduces the steganography channel in
  the one term most prone to it. Worth watching in run #2.
* `scripts/inspect_phase1.py` now scores against the pooled target, so its
  baselines stay comparable to the training log, and prints the pool factor.

**Predicted effect.** On a synthetic mix matching the measured 5.3%/94.7%
split, pooling by 16 moves the headroom between the two trivial predictors from
**5.2% → 46.5%**. That is an upper bound: it assumes the within-window component
is white, and MERT frames are temporally correlated, so the real gain will be
smaller. `inspect_phase1.py` measures the true value on run #2.

**Also fixed** (was an open item, and it blocked local verification): the eager
`src.models.flow` import in `src/models/__init__.py` is now lazy via a module
`__getattr__`. Phase-1 code no longer needs zuko installed; the flow names still
resolve on first use.

**Verification** — `scripts/smoke_test_model.py` passes (90 → 5 frames, all five
losses finite, backward runs, layer-mix grads present); pooling is a true block
mean, a no-op at factor 1, and drops the trailing partial block; `--recon-pool`
round-trips through `apply_overrides` and the checkpoint config.

**Not changed:** `cosine_weight` is still 0.0. Turning it on is a separate knob
and a separate ablation.

---

## Open items

Carried forward, not yet acted on:

* `scripts/train.py` should reject an empty `--data-root` instead of resolving it
  to `.`.
* `mil_nce` does not mask same-`song_id` negatives. Harmless for Phase 1, but it
  caps the achievable contrastive loss and must be fixed before any real
  representation-learning run.
* The contrastive number (0.36 vs. chance 1.386) is suggestive but weak evidence:
  4-way discrimination is easy, and `n_candidates=1` means MIL-NCE degenerates to
  plain InfoNCE. The lever for a stronger signal is `--n-candidates`, not a larger
  batch.
* `.gitignore` has a `checkpoionts/` typo, so `checkpoints/` is not actually
  ignored, plus stray `IGNORE` lines.
* `extract_mixes`' docstring overstates what micro-batching achieves (see bug 5
  above).
