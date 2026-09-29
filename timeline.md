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

## 2026-09-18 — Data audit and the learning-curve pipeline

### Where the data actually stands

Measured from `data/logs/*_downloaded_songs.csv` and the phase-1 alignments:

| | tracks | works (compositions) |
|---|---|---|
| train, de-contaminated, in the CSV | 96,195 | 1,639 |
| train, downloaded | 49,809 | 931 |
| phase-1 aligned set (= the planned phase-2 cache) | 4,010 | **105** |

The downloader went work by work, so the missing train tracks all belong to
708 works with nothing downloaded. At 76 % yield the real train ceiling is
~81 k tracks, not 100 k. The binding axis for content (contrastive labels,
the conditioning of p(style | content)) is the number of **works**, and
training had seen 105 of the 931 on disk.

Trainable parameters: the representation model has 127.7 M (encoders
2 × 50.4 M, bottlenecks 2 × 4.5 M, decoder 18.0 M). The flows have 237.4 M
(style 174.0 M, content 63.5 M), more than the encoder stack.

### Decisions

1. **A fixed number of usable pairs per work** (`align_covers.py
   --kept-per-song 20`). Candidates are walked in round-robin order: each round
   is a perfect matching, so the 20 pairs cover as many distinct recordings as
   possible. Works align in parallel rounds until 20 pairs score ≥ 0.2, or
   until 200 candidates are spent. The choice is written to
   `alignments/{split}/selection_k20.json`, and `build_manifest.py
   --kept-per-song 20` reads that file, never recomputing the order. The order
   depends on the exact set of chroma files, so recomputing it would silently
   disagree the moment one track differed.
   Uncapped, the largest work (1,926 downloaded versions) would own 37 % of
   all 4.96 M candidate pairs.
2. **Designated eval splits `val50` / `test50`** (`splits/eval_works.json`,
   committed). The pool is the official val ∪ test works with ≥ 10 downloaded
   tracks, which is exactly 100 works. Those are split into size-matched halves
   by seeded coin flips over consecutive pairs sorted by clique size. Carving
   from train was rejected because compositions are the scarce resource, and
   the official held-out works are already guaranteed disjoint from train. The
   2 works listed in both official splits (5854, 186755) hold all 76 shared
   videos and were downloaded under test only. They are sourced from one split,
   so no video can land on both sides. Eval manifests cap tracks at 40 per work.
   Without the cap, one work would hold 29 % of test50's tracks.
3. **The learning curve runs online (MERT live, layer mixes frozen), not from the
   phase-2 cache.** The cache for all 931 works would be ~0.9 TB, and the curve
   is what decides how many works the cache needs to hold. It would also save
   little: most of the 1.31 s step is spent in the encoders, not MERT.
4. **Protocol.** Three nested, size-stratified work subsets (25/50/100 %,
   `subset_manifest.py`). All three get the same step budget (≈ 10 epochs of the
   full pair set) and the same frozen mixes, loaded by `train.py
   --layer-weights`. Each keeps `best.pt` on val50 same-work mAP, is scored once
   on test50, and the runs are compared by a **paired** bootstrap over test
   works on identical windows. The runbook is
   `docs/learning_curve_runbook.md`.
5. **`mil_nce` now masks same-work negatives** (was an open item). This mattered
   for the learning curve itself: the smaller the subset, the more often a batch
   holds two pairs of one work. Unmasked, w25 would have faced ~4× the false
   negatives of w100.

### Found while testing

* **The val loader iterated pairs in manifest order**, so a batch of 4 held one
  work. After the mask, every negative dropped and `val/contrastive` read
  exactly 0.0000. Before the mask, it measured mostly false negatives. The val
  loader now uses a fixed seeded permutation: works mix within a batch, and the
  order is identical at every validation and across runs with the same seed.
* **Alignment and chroma writes are now atomic** (write `.part`, then rename),
  and the target mode deletes and redoes unreadable `.npz` files. A spot reclaim
  mid-write can no longer leave a truncated file that resume logic trusts.

### Verification

End to end on a synthetic corpus: real SHS keys, generated audio in which
covers share a melody at different tempo and key and every fourth "cover" is
unrelated, real chroma and alignment, and a stub MERT. The stages were target
alignment (resume computes 0 and reproduces the selection), train / val50 /
test50 manifests (paths from both source splits, the cap respected), nested
subsets, `train.py` on w25 and w100 (frozen mixes, validation every N steps,
`best.pt`, `val_metrics.jsonl`), `evaluate_encoder.py` with bootstrap intervals,
and `learning_curve.py` with paired differences.

Unit-checked:

* the MIL-NCE mask against a hand computation, for K = 1 and K = 2
* same-recording exclusion in the retrieval metric
* the real `phase1_layer_weights.pt`, which loads into the full model (content
  peaks at layer 9, style at 10); its standardizer stats are skipped when
  `recon_pool` differs
* subset exactness at the real scale (233 / 465 of 931 works)
* `smoke_test_model.py`, which still passes

---

## 2026-09-27 — Learning-curve run 1 of 3 (`lc-w25`): the content encoder learns, then memorises

First run of the 2026-09-18 protocol. It answers more than it was designed to,
because it overfits visibly and the trajectory says where and why.

**Configuration.** Measured from `run_info.json` (`q-w25`, which uses the same
`train_w25` manifest) and the run header, 2026-09-29.

| | |
|---|---|
| instance | g5.2xlarge (A10G) |
| kept pairs per work | **40**, not the 20 the runbook specifies |
| `train_w25` | **241 works** (239 of them contributing pairs), **8,744 pairs**, 13,273 tracks |
| steps/epoch | 2,186 → 40 epochs for the 87,372-step budget |
| `train_w100` (sets the budget) | 34,949 pairs → 87,372 steps |
| epochs | **40** — the budget is fixed in steps, so w25 makes 4× the passes w100 will |
| batch | `--batch-pairs 4 --batch-tracks 4`, `n_candidates = 1` |
| `val50` | **1,703 pairs** (recovered from the retrieval denominators: every value is k/1703) |
| layer mixes | loaded from `phase1_layer_weights.pt` and frozen |
| selection | `best.pt` on `val/content_work_map` |

### Held-out content retrieval, measured for the first time

| step | val work mAP | val work R@1 | val pair R@1 |
|---|---|---|---|
| 2,500 | 0.112 | 0.142 | 0.008 |
| 12,500 | 0.155 | 0.219 | 0.006 |
| 22,500 | 0.174 | 0.235 | 0.008 |
| 50,000 | 0.184 | 0.232 | 0.012 |
| **67,500** | **0.188** ← peak, `best.pt` | 0.262 | 0.013 |
| 87,372 | 0.185 | 0.255 | 0.010 |

Chance on a 1,703-query pool is 1/1703 = 0.059 % for pair R@1 and ≈ 2 % for the
work metrics (≈ 34 same-work candidates per query if val50's 1,703 pairs spread
evenly over its 50 works). So work mAP is ~9× chance and pair R@1 ~17–25×
chance: **real content invariance across unseen compositions, and modest.**

**The last 23 % of the budget bought nothing.** The metric is flat from ~step
60,000 (0.184–0.188, oscillating) to the end.

### It overfits, and it starts at epoch 7

| step | train/contrastive | val/contrastive | gap |
|---|---|---|---|
| 2,500 | 1.087 | 0.930 | −0.157 |
| 15,000 | 0.724 | **0.758** ← val minimum | +0.034 |
| 50,000 | 0.522 | 0.921 | +0.399 |
| 87,372 | 0.329 | 0.947 | +0.618 |

Train falls monotonically; val bottoms at step 15,000 (epoch ~7, i.e. seven
passes over the pair set) and climbs steadily after.

**Only the contrastive term overfits.** `val/recon` falls monotonically all run
(1.926 → 1.818) and `val/swap` with it. That is the 2026-09-18 prediction
confirmed: reconstruction is per-recording, and w25 still holds thousands of
recordings, so works are not its scarce resource. `val/cycle` falls
0.071 → 0.012 and `val/decorrelation` 0.125 → 0.103.

### Why it overfits

The contrastive task as configured is *solvable by memorising work identity*,
and four things make that the path of least resistance:

1. **241 works.** "Pick the cover among 4" needs only a work-identity embedding
   for each of the 13,273 recordings (8,744 pairs over 239 of those works).
   That generalises to zero unseen works, which is exactly what the held-out
   plateau shows.
2. **40 passes over the pair set.** Anchors are resampled per visit, so the
   *windows* differ, but the recording pair repeats 40 times — which is what
   memorisation needs. Overfitting begins at pass 7.
3. **Three negatives.** `n_candidates = 1` at `batch_pairs 4` makes each step a
   4-way choice. Once works are coarsely separated the loss saturates (many
   steps at ~0.05, against chance ln 4 = 1.386), gradients collapse, and the
   remaining 70,000 steps sharpen work-specific features instead of general
   ones. This is the mechanism that converts a saturated task into memorisation,
   and it is the already-open `--n-candidates` item below.
4. **No regularisation on the content path.** `EncoderConfig.dropout = 0.0`,
   `weight_decay = 0.01`, 50.4 M parameters per encoder.

Note the loss and the metric disagree about *when* things go wrong:
`val/contrastive` degrades from step 15,000 while `val/content_work_map` keeps
improving to 67,500. At temperature 0.1 the loss punishes *confident* errors, so
it tracks calibration; mAP tracks ranking. **Selecting on
`val/content_work_map` was load-bearing** — `val/total` would have picked
~step 32,500 and thrown away real gains.

### The reconstruction ceiling, quantified

`inspect_phase1.py` on `last.pt`, against `val50`:

```
predict dataset mean      2.0230
MODEL recon               1.8565
predict per-window mean   1.7897     <- the model is still worse than this
```

71.4 % of the between-window range, 8.2 % of total variance. This confirms
run #2's 61 % rather than contradicting it, and adds the number that was
missing: **a per-window constant explains only 11.5 % of the pooled target's
variance at `recon_pool = 16`.** So 88.5 % of what the decoder is scored on is
still temporal detail inside the window that a 16×256 latent set cannot carry.
Pooling by 16 was not enough; run #2 already called a larger `recon_pool` "the
obvious lever" and this prices it.

Decoder output std is 0.2901, and 0.2901² = 0.084 against 0.082 variance
explained. Those matching **is** MSE-optimal shrinkage under a mostly
unpredictable target — not steganography, and not a capacity limit. Growing the
decoder remains ruled out.

### The frozen layer mixes are 99.7 % the same vector

The checkpoint's mixes are bit-identical to `phase1_layer_weights.pt`
(cosine 1.000000, max abs difference 5e-5), so `--freeze-layer-weights` did what
it says and `inspect_phase1.py` here reports phase 1's result, not drift.

| | content | style |
|---|---|---|
| cos to uniform | 0.9866 | 0.9942 |
| max/min weight | 1.69 | 1.40 |

`cos(content, style) = 0.9972`.

The run-#2 freeze met the criterion **as written** — style's deviation from
uniform had grown 7.2× and was annealing monotonically — and the criterion sets
no numeric threshold. But in absolute terms both mixes sit within ~1 % cosine of
uniform and are 99.7 % identical to each other. **That effectively answers the
"cache one stream or two" question left open on 2026-09-01: one stream.** Two
near-duplicate streams cannot justify 2× the phase-2 storage, and the honest
thesis finding is that the per-branch layer mix did not earn its keep.

### Decisions

1. **Keep 40 kept pairs per work for w50 and w100.** Changing it now would
   confound the curve — the design varies works and holds pairs-per-work fixed.
   40 is also the better setting on this run's own evidence: it gives ~70 %
   recording coverage against ~52 % at 20, and because the step budget is fixed,
   more pairs mean *fewer passes* over each one (w100 will make ~10 against
   w25's 40), which directly attacks cause 2 above.
2. **Finish the curve before fixing anything.** The recon ceiling and the layer
   mixes are on the reconstruction/style side; the overfitting is on the
   contrastive/content side. w50 and w100 measure the slope in works, which is
   what decides whether the 708 undownloaded works are worth fetching, and that
   conclusion survives a later `recon_pool` change.
3. **Do not extend the step budget.** 87,372 steps is ~23 % longer than useful
   at w25. Keep it for comparability, and expect w100's `best.pt` to land later
   in the run if works are the binding constraint.
4. **Treat the negative queue and more works as fixes for different symptoms.**
   More negatives attacks the *plateau* (a saturated 4-way task). More works
   attacks the *train/val gap*. A queue does not add works, and with 241 works a
   memorised work classifier still solves a 1,000-negative task — so it cannot
   substitute for works.

### Tooling added

* **`scripts/diagnose_training.py`** — blocks a `train.log`'s per-step samples,
  averages, and bootstraps the first-vs-last difference, because at 4 pairs a
  single step's contrastive value swings ~1.7 regardless of progress. Validated
  against synthetic logs of a learning run and a flat one.
* **`train.py --overfit-batches N`** — trains on N fixed batches with validation
  off, reseeding the window RNG each epoch so the *same* audio replays
  (`windows.py` draws a fresh anchor per access, so without this "fixed batches"
  would hold new windows each epoch). Losses that refuse to collapse indicate
  capacity or optimisation rather than data.
* **`inspect_phase1.py`** — the standardiser and steganography notes were
  unconditional legend text that read as findings; both are now conditional, and
  the "a per-window constant explains X % of this target" line was added, which
  is the number that prices `recon_pool`. `--recon-pool` already existed, so the
  pooling sweep needs no retraining to *measure*.
* **`align_covers.py` now line-buffers stdout.** Piping to `tee` makes stdout a
  pipe, which Python block-buffers at ~8 KB; the train chroma stage printed
  nothing for hours and looked hung while working normally. The chroma progress
  line now carries a rate and an ETA.

---

## 2026-09-27 — Two fixes, and why the learning curve had to be restructured

`lc-w25` produced two distinct symptoms with two distinct causes (previous
entry): a **plateau** from step ~60 k, caused by a saturated 4-way contrastive
task, and a widening **train/val gap** from pass 7, caused by work memorisation
over 241 works. Both are now addressed, and addressing them invalidated the
experiment they were diagnosed from.

### Fix 1 — contrastive negative queue

`ContentQueue` in `src/losses.py`: a FIFO of recent pooled content vectors that
`mil_nce` appends to the denominator. `--negative-queue N`, **default 1000**,
`0` = off.

* Only the **B-side candidates** are queued, tagged with their work, because
  the B candidates are what populate the denominator.
* **Queue entries of the anchor's own work are masked out**, exactly like
  in-batch ones. Without that the queue would reintroduce the false negatives
  the `mil_nce` mask exists to remove — at 241 works a 1,000-entry queue holds
  ~4 entries of the anchor's own work at any moment.
* **Buffers are non-persistent**, so the queue never enters a checkpoint: older
  checkpoints still load, and a resumed run refills over ~250 steps.
* **Validation does not use the queue**, so `val/contrastive` stays comparable
  with the existing `val_metrics.jsonl` history.
* Entries are detached and go stale as the encoder drifts. At 1,000 entries and
  4 pushed per step that is 250 steps (~5 min) of history. Much larger would
  want a momentum encoder (MoCo), which costs a second copy of the weights.

Verified against a hand-computed loss to 1e-5; same-work masking leaves the
affected row bit-identical; the FIFO evicts correctly and handles a batch larger
than itself; entries are stored L2-normalised; `state_dict()` is empty; loss
rises monotonically from 0 to 512 queued negatives.

### Fix 2 — style-only augmentation

`src/data/augment.py`: random EQ tilt, a resonant peak, a codec-like lowpass, a
cheap reverb, tanh saturation, additive noise at a target SNR, and gain. All FFT
or elementwise in pure torch — torchaudio is not installed on the DLAMI, and an
IIR filter in Python would be far slower than filtering a spectrum. Loudness is
renormalised before the gain step, or the level itself becomes a cue the encoder
reads instead of the timbre change. `--augment`, **off by default**.

**Pitch and tempo are deliberately not augmented.** Real covers already supply
those, and `CLAUDE.md` wants key invariance to come from real transpositions.
What remains is exactly the content/style split the project is built on, so the
content encoder *should* be invariant to all of it.

**Augmented windows never become reconstruction targets.** Augmentation applies
to the pair's A-side window only, and `compute_losses` takes a new `recon_index`
that excludes those windows from term 2a and from the Standardizer update.
Fitting the decoder and the style branch to augmented audio would teach
P(style | content) that lowpassed, saturated, reverberant signal is ordinary
human style — which is the distribution the detector later scores against. The
swap term is untouched, since it targets B.

`scripts/check_augment.py` measures content preservation with the project's own
metric (beat-synchronous chroma → OTI → Smith-Waterman) rather than assuming it.
On synthetic music:

| transform | score vs the clean self-alignment |
|---|---|
| gain, EQ tilt, EQ peak, lowpass, saturation | 100.0 % |
| noise | 99.5 % |
| **reverb** | **76.4 %** |
| all defaults together | 73.5 % |

Reverb smears transients, which blurs beat tracking and chroma together. Even so
0.735 is far above the 0.2 keep threshold, so the aligner still calls it the
same content. Re-measure on real audio before trusting it.

**Found while testing:** augmentation originally drew from the global `random`
stream, which shifted every later window draw — so an augment-on and an
augment-off run sampled *different windows* and the ablation would have been
confounded. It now uses `random.Random(torch.initial_seed() + index)`: same
windows, same anchors, only the A waveform differs.

### The learning curve had to be restructured

A curve measures d(metric)/d(works) **under one configuration**. Both fixes
change that derivative, in opposite directions:

* **Augmentation substitutes for works.** It synthesises extra performances of a
  work the model already has — the same resource the curve varies — so it should
  **flatten** the curve.
* **The queue may let the model exploit more works.** A model that keeps
  receiving gradient about fine distinctions can use diversity a saturated one
  ignores, so it should **steepen** it.

Which dominates is not predictable, and the decision — fetch the remaining 708
works or not — depends on the slope at the top end **under the configuration
that ships**. Measuring it under the old configuration risks over-buying data
(if augmentation would have sufficed) or under-buying it (if the fixes unlock
more). There is also a plain accounting objection: a curve anchored on `lc-w25`
measures how much more data helps a model that is broken in two known ways.

**DECISION: two stages, each varying one thing.** `docs/learning_curve_runbook.md`
is rewritten around it.

| stage | runs | varies | holds fixed |
|---|---|---|---|
| **A** | `q-w25`, `qa-w25` (+ `lc-w25`, done) | the configuration | data (`train_w25`) |
| **B** | `w50`, `w100` at A's winner | the data | the configuration |

Four runs total, ~$90, ~2 days on two boxes at a time; `lc-w25` is stage A's
third arm and stage A's winner is stage B's 25 % point, so neither is re-run.

* **Everything keeps `lc-w25`'s 87,372-step budget and K = 40**, or nothing is
  comparable.
* **Stage A runs the full budget, not a cheap short version.** Ranking the
  configurations in 25 k steps would cost a third as much — `lc-w25` was already
  at 0.174 mAP by step 22,500 — but the queue's entire purpose is to prevent the
  *late* saturation, so a short run would systematically under-measure it.
* **The train/val contrastive gap is stage A's load-bearing number**, not just
  the mAP. `lc-w25` ended at +0.618. A much smaller gap means memorisation was
  genuinely reduced, which is also a free prediction that stage B's curve will
  be flatter.
* **Stage B's conclusion is conditional** and must be written that way: "at the
  configuration we ship, works saturate at N", not "works saturate at N".
* Three points give a slope per doubling. Extrapolate to 1,639 works (all
  de-contaminated train works) and to ~10 k (SHS100K-v2 scale) before deciding;
  Da-TACOS is ~1,000 works and SHS100K-v2 ~10 k, so ~970 is small for a
  contrastive problem and a flat w50 → w100 step is evidence about this
  configuration's appetite, not proof that data stopped mattering.

**Footgun introduced:** `--negative-queue` defaults to 1000, so `lc-w25`'s
configuration is no longer what a bare command produces. Reproducing or resuming
it needs an explicit `--negative-queue 0`. `run_info.json` now records
`negative_queue`, `n_candidates` and `augment`, so runs are self-describing.
Also `train/contrastive` is no longer comparable across stage A — with a queue
it is a ~1,000-way loss instead of 4-way.

---

## 2026-09-29 — Runs must not depend on shell state

`launch qa-w25 train_w25 --negative-queue 1000 --augment` started, printed
`data_root=/home/ubuntu/Tesis/data` and died on a missing
`manifests/train_w25/tracks.jsonl`. The manifests were fine; the shell was not.
`q-w25` had been launched from a pane where `$COMMON` and the `launch` function
were defined, and the second pane had neither, so `--data-root`, `--val-split`,
`--layer-weights`, `--freeze-layer-weights` and `--max-steps` were all silently
absent and `train.py` fell back to the repo's own `data/`.

This is box B blocker 1 in a new costume: **tmux panes do not inherit shell
state defined in another pane.** The dangerous part is not that it failed — it
is the failure mode of a run that *doesn't* fail. Had `~/Tesis/data/manifests`
happened to contain a `train_w25`, the run would have trained on the wrong data,
unfrozen the layer mixes, validated on the wrong split and stopped after the
default 10 epochs, and nothing in the log would have said so.

**Three fixes:**

1. **`scripts/launch_run.sh`** (committed) holds everything that must be
   identical across runs. It refuses to start without `$DATA`, checks every
   manifest it needs and lists what is present when one is missing, derives the
   budget from `train_w100`'s pair count so the number does not depend on the
   split being launched, and echoes `run=… split=… steps=… data=…` before
   starting. `steps=87372` must appear or nothing is comparable to `lc-w25`.
   It also runs `python -u` (the chroma buffering lesson) and `tee -a`, so a
   resumed run appends to `train.log` instead of truncating it.
2. **`train.py` fails with the resolved root in the message** when
   `manifests/{train_split}/tracks.jsonl` is absent, names the likely cause when
   `--data-root` was not passed at all, and lists the splits that *are* present.
3. **`--data-root` is parsed as a string, not a `Path`** — closing the open item
   from box B blocker 2. `Path("")` collapses to `"."`, so after argparse an
   empty `$DATA` was indistinguishable from an explicit current directory; as a
   string it is caught with a message that names `$DATA`.

Verified: `$DATA` unset, `$DATA` set but manifests absent, too few arguments,
`--data-root ""`, `--data-root` omitted entirely, and that the budget resolves
to 87,372 whether `train_w25` or `train_w100` is launched.

---

## Open items

Carried forward, not yet acted on:

* ~~`scripts/train.py` should reject an empty `--data-root` instead of resolving
  it to `.`.~~ Fixed 2026-09-29: parsed as a string so `Path("")` cannot hide
  it, plus a manifest-existence check that names the resolved root.
* ~~`mil_nce` does not mask same-`song_id` negatives.~~ Fixed 2026-09-18
  (`groups` argument; `train.py` passes each pair's work id).
* ~~The contrastive number is weak evidence: 4-way discrimination is easy.~~
  **Confirmed by measurement** 2026-09-27: at `batch_pairs 4` / `n_candidates 1`
  each anchor gets 3 negatives, the loss saturates, and the plateau follows. The
  lever is more negatives. **Implemented** 2026-09-27 as `ContentQueue`
  (`--negative-queue`, default 1000); `--n-candidates` remains untried and is
  complementary.
* **Sweep `recon_pool`.** At 16, a per-window constant explains only 11.5 % of
  the pooled target's variance, so the objective is ~9/10 irreducible. Measure
  the share at 16/32/64/96 with `inspect_phase1.py --recon-pool` (no retraining
  needed for the baselines) before picking a new value.
* **Nothing regularises the content path**: `EncoderConfig.dropout = 0.0`,
  `weight_decay = 0.01`, 50.4 M parameters per encoder, 241 works in w25.
  Augmentation is now implemented (`--augment`) and keeps off the
  reconstruction target; **dropout is still 0.0 and untried**, and is the
  cheapest remaining lever.
* **`--kept-per-song` could be work-aware.** A flat K spends the budget badly at
  both ends: `clamp(n_w, 20, 100)` reaches ~84 % recording coverage against 70 %
  at flat 40, still keeps the largest work at 0.24 % of all pairs, and needs
  ~100 k candidate alignments (~0.3 core-hours). Deferred until the curve is in,
  because changing it mid-curve would confound it.
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
* Validation runs in fp32 (no autocast). That is harmless, but val50 turned out
  to hold 1,703 pairs, so a pass is ~426 batches. `--val-max-batches` is the
  lever if it matters; the val order is a fixed permutation, so a capped pass is
  the same subset every time. Note the retrieval metrics pool the *whole* pass,
  so a cap makes the task easier and must be held constant across comparisons.
* The learning-curve intervals cover test-set sampling, not training seeds. A
  borderline w50 → w100 call needs a second seed of w100.
* **Queue entries go stale.** They are detached and never re-encoded, so at
  much more than ~1,000 entries a momentum encoder (MoCo) would be needed. The
  current default is sized so the queue holds ~250 steps of history.
* **`check_augment.py` has only been run on synthetic audio.** Re-measure on
  real tracks before trusting the 76 % reverb figure, and turn `reverb_seconds`
  down if it is worse there.
* ~~Phase 2: cache one stream or two?~~ Effectively answered 2026-09-27:
  `cos(content, style) = 0.9972` and both mixes are within ~1 % cosine of
  uniform, so **one stream**. The per-batch sequence correlation that
  `extract_mert_features.py` reports is still worth reading on real pairs before
  the cache is built, but only as a confirmation.
