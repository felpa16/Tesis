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

## 2026-08-29 — Phase-1 run #2 (pooled target) and the decision to freeze

Same configuration as run #1 (7,469 tracks, 4+4 batch, 20 s windows, 10 epochs,
21,420 steps) with `--recon-pool 16 --checkpoint-dir checkpoints/run2-pool16`.

**Terminal losses**

```
recon 1.9085   contrastive 0.1660   swap 1.7909   cycle 0.0374   decorrelation 0.0557
```

**Reconstruction against the trivial predictors** (`recon_pool=16`, 93-frame target)

| | run #1 | run #2 |
|---|---|---|
| dataset-mean baseline | 1.9526 | 2.0413 |
| model `recon` | 1.9348 | 1.8851 |
| per-window-mean floor | 1.8499 | 1.7851 |
| **share of between-window range** | **17.3%** | **61.0%** |
| decoder output std (target 1.0) | 0.085 | 0.279 |

The absolute loss barely moved (1.93 -> 1.89) because **87% of the pooled
target's variance is still within-window**. The between-window headroom went
5.26% -> 12.55% of total variance — real, but far below the 46.5% the
white-noise estimate predicted. MERT frames are strongly temporally correlated,
so pooling 16 frames cut the within-window component by only ~2.4x, not 16x. A
larger `recon_pool` is the obvious lever and costs nothing to try, but it can be
tuned in phase 2 without re-running MERT.

`cycle` rose 0.0025 -> 0.0374 (~15x), as intended: decoding at pooled resolution
removes the unconstrained high-frequency channel that made the decode->re-encode
path trivially invertible.

### Layer mix

| | run #1 | run #2 |
|---|---|---|
| content ‖deviation from uniform‖ | 0.0328 | 0.0330 |
| style ‖deviation from uniform‖ | **0.0030** | **0.0217** |
| style cos-to-uniform | 0.999884 | 0.994195 |
| style max/min weight | 1.06x | 1.40x |

**The content vector reproduced across two materially different objectives:
cosine 0.9976 between run #1 and run #2 deviations**, same peak layers (7-10),
same contrast. That is the strongest available evidence it reflects a property of
MERT rather than an optimisation artifact.

**Style woke up — 7.2x more contrast.** Its run-#1 shape (peaking at layers 2-5)
is not a competing measurement that run #2 overturned; at ‖dev‖ = 0.0030 it was a
parameter that had never moved.

### The per-epoch trajectories are what settled it

Rotation of each weight vector per epoch, in degrees:

```
              e1    e2    e3    e4    e5    e6    e7    e8    e9
run1 content 2.01  1.43  1.00  0.68  0.44  0.26  0.18  0.08  0.08   monotone
run1 style   0.14  0.00  0.00  0.14  0.24  0.24  0.21  0.14  0.08   NOT monotone
run2 content 2.08  1.48  0.99  0.62  0.53  0.24  0.11  0.08  0.08   monotone
run2 style   0.96  0.83  0.67  0.46  0.39  0.37  0.27  0.20  0.11   monotone
```

Run #1's style vector *wobbles* — it reads perfectly converged (0.00 degrees) at
epochs 2-3 and then moves again. That is random drift in a parameter receiving no
useful gradient, not annealing. Run #2's style anneals monotonically from 0.96
to 0.11 degrees, the same shape as content.

Cross-check: style's first-epoch movement grew **6.8x** from run #1 to run #2,
independently matching the **7.2x** growth measured in the final vectors'
deviation norms.

**Methodological finding.** `log_layer_weights` computes the cosine on the
softmax vector, which is dominated by its uniform component — so a vector pinned
at uniform scores ~1.000000 and looks *more* converged than one that is genuinely
learning. In run #1 style read 0.999997 at epoch 1 while content read 0.999382.
**The freeze metric as implemented is structurally blind to the exact failure it
exists to catch.** The sensitive version is the cosine between successive
*deviations from uniform*. This matters for phase 2's planned sanity ablation
(resume online training and confirm the frozen weights do not want to drift),
which relies on this same metric.

### DECISION: freeze the layer weights

All three conditions of the amended criterion (`CLAUDE.md`, Feature caching) are
met:

1. **Stationarity** — both branches monotone, final epoch ~0.1 degrees.
2. **Departure from uniform** — content 0.0330, style 0.0217 (was 0.0030).
3. **Driving objective learning** — `recon` at 61% of the between-window range
   (was 17%); contrastive at 0.166 against a chance level of ln(4) = 1.386.

Freezing now is safe because the 50 weights are the *only* phase-1 output.
`recon_pool`, `cosine_weight`, decoder capacity and the loss weights are all
tunable in phase 2 without re-running MERT, since none of them change what gets
cached. A third phase-1 run would spend 8 hours refining something phase 1 does
not produce.

### Open question for phase 2: cache one stream or two?

The two branches converged on nearly the same layers. Cosine between their
deviations went **0.39 (run #1) -> 0.93 (run #2)**; on the full vectors,
`cos(content, style) = 0.997`. Style's only gradient is reconstruction,
reconstruction wants maximum information, and MERT's mid-layers are the most
information-dense — so style migrated to where content already was.

This is not a failure, but `CLAUDE.md`'s ~7 TB phase-2 cache estimate assumes two
distinct streams. **Before committing to the cache, measure the correlation
between the two resulting mix *sequences*** (not the weight vectors — the hidden
states are already highly correlated across layers, so the sequences will be
closer than 0.997 suggests). Above ~0.99, cache one stream and feed both encoders
from it: ~3.5 TB saved, and a legitimate thesis finding that the per-branch layer
mix did not earn its keep. It should be a deliberate call, not a discovery made
after paying for the storage.

### Artifacts

Saved to S3 before terminating the instance: the run-#2 checkpoint, an extracted
`phase1_layer_weights.pt` (both weight vectors plus the standardiser buffers),
the TensorBoard event files for both runs, and the manifests.

---

## 2026-09-01 — Phase-2 feature extraction: the cached unit is the window

`scripts/extract_mert_features.py` materializes the phase-2 training set. It
walks `manifests/{split}/pairs.jsonl`, samples aligned window pairs through each
pair's warping path the way `AlignedPairDataset` does, runs frozen MERT on each
window, applies the two softmax vectors from
`checkpoints/run2-pool16/phase1_layer_weights.pt`, and uploads

    s3://BUCKET/mert-features/{content,style}/{split}/{window_id}.npy   (N,1024) fp16

`window_id` is `{track_key}_{start_ms:08d}`, so an object names the track and the
exact offset it came from. Which windows form a pair is recorded in
`manifests/{split}/window_pairs.jsonl` (`WindowPairEntry` in
`src/data/manifest.py`), written locally and uploaded next to the features; that
file is the index phase-2 training reads.

### Why the window and not the track

The first cut of this script cached whole tracks and assumed windows could be
sliced out of the cached stream later. That is wrong, and the reason is worth
keeping: **MERT features are not a local function of the audio.** Two
independent mechanisms, both measured on this model:

* **The conv frontend is `HubertGroupNormConvLayer`: `GroupNorm(groups=512,
  channels=512)`.** One group per channel means each channel is normalised over
  the *entire* time axis of whatever you feed it. Running the frontend on a 10 s
  clip versus the same 10 s sitting inside a 30 s clip moves the output by a
  per-frame cosine of **0.93** — before a single attention layer runs.
* **The encoder is global, and context margins do not fix it.** Discarding
  1 s / 2.5 s / 5 s of context per side gives last-layer cosines of
  **0.85 / 0.88 / 0.92** against an unchunked forward. It converges toward 1
  only as the margin approaches the whole input.

So no track-level cache can be sliced into windows without changing the
features, and a whole-track forward is not a meaningful target anyway: MERT was
pretrained on ~5 s clips, and attention over 22,500 frames is both out of
distribution and quadratic.

Caching the *window* sidesteps all of it. Each cached object is one
`MERTModel.forward` on exactly the waveform `decode_window` produces, so it is
bit-for-bit what `scripts/train.py` computes online for that window. The
segment-length question disappears because S is fixed at extraction time.

### Design decisions

* **Anchors are deterministic** — evenly spaced over the alignment points whose
  windows fit inside *both* covers, rather than drawn at random per epoch. A
  rerun reproduces the training set instead of resampling it, and the windows
  cover the aligned span instead of clustering wherever the RNG landed.
* **Windows are deduplicated across pairs.** A cover aligned against two others
  often lands on the same offset twice; it is computed and stored once, and the
  manifest references the id twice. In the fixture, 3 pairs x 2 windows = 12
  slots collapsed to 8 unique windows.
* **A window counts as cached only when both streams are in S3**, so a
  half-finished upload is redone rather than silently leaving a one-sided
  sample.
* **`--s3-audio` streams the corpus instead of requiring it on the instance.**
  ffmpeg needs a seekable input to cut a window (`-ss` before `-i`), so an S3
  object cannot simply be piped through stdin. Work is therefore grouped by
  source track: each track is pulled to a temp file once, every window it owes
  is cut from it, and it is deleted — so transfer is one copy of each track in
  the pair set (~4 MB, same-region and free), and disk high-water is
  `--decode-workers` files, not the corpus. Sharding is by track group for the
  same reason: two shards must never fetch the same file. Manifests and
  alignments stay local — 88 MB for train's 20,459 `.npz`, so syncing them is
  not the problem the audio is.
* Resumable and shardable across GPUs; every shard writes the same manifest, so
  training sees one coherent index no matter how the work was split.

### Verified before shipping

* `feature_extractor -> feature_projection -> encoder` reproduces
  `MERTModel.forward` to **0.0** max abs difference — the staged decomposition
  used to diagnose the GroupNorm behaviour above is sound.
* A cached window equals `decode_window` + an online forward to within fp16
  storage: max abs diff 0.062, **0.83% of a std** (the fp16 quantisation floor).
* Batching windows into one forward is numerically neutral: per-frame cosine
  0.9999999, differences at one fp16 ULP.
* Frame accounting: 20 s -> 1,499 frames; 10 s -> 749.

### Storage, and the two streams

One 20 s window is 1,499 x 1024 fp16 = **3.07 MB per stream**, so a pair sample
(two windows, two streams) is ~12.3 MB. The 8,570 surviving train pairs at
`--windows-per-pair 4` come to **~420 GB** — an order of magnitude under the
~7 TB the whole-track plan implied, because nothing between the sampled windows
is stored.

The script reports, per batch, the frame-wise cosine between the content mix and
the style mix, raw and after subtracting each window's own mean frame. This is
the measurement the run-#2 entry asked for before paying for the cache. On probe
audio it is already **0.9999 centered**, which is what the near-uniform weight
vectors (`cos(content, style) = 0.997`, max/min weight ratio 1.67 and 1.40)
predict. Read it on the first ~50 real pairs with `--limit` before committing to
two streams: above ~0.99 centered, cache one and halve the storage.

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
* `log_layer_weights` should also log the cosine between successive deviations
  from uniform — the plain softmax cosine is insensitive (see run #2 entry).
* `.gitignore`'s `checkpoionts/` typo is fixed; the stray `IGNORE` lines remain.
* Materializing windows fixes the anchors, so phase-2 loses the per-epoch
  resampling that online training got for free. `--windows-per-pair` is the
  dial; whether fixed anchors cost anything measurable against online sampling
  is untested.
* The plain-reconstruction stream (`batch_tracks`, `TrackWindowDataset`) has no
  cached equivalent yet — only aligned pair windows are materialized. Pair
  windows do feed loss 2a, so this is a diversity question, not a blocker.
* `extract_mixes`' docstring overstates what micro-batching achieves (see bug 5
  above).
