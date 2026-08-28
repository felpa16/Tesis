# Synthetic music detection via content–style disentanglement

A research prototype that detects AI-generated music **without ever training on
AI-generated music**.

Conventional detectors learn to recognize the generators they were shown, and
tend to fail on the next one. This project instead learns the distribution of
*human* music in a disentangled latent space, and flags anything that falls
outside it. A song is decomposed into

* **content** — melody, harmony, lyrics, chord progression, musical form
* **style** — instrumentation, timbre, expressive timing, dynamics, production

and the detector models the conditional distribution `P(style | content)` of
human performances. The central hypothesis is that current generators reproduce
convincing content while sampling style badly:

```
P_human(style | content)  ≠  P_AI(style | content)
```

Supervision for the disentanglement comes from **cover songs**: two covers of
one composition share content but differ in style. That is why the dataset is
SHS100K, and why so much of this repository is alignment machinery.

The design rationale, hyperparameter choices and open research questions live in
[`CLAUDE.md`](CLAUDE.md). This file is about operating the code.

---

## Pipeline

```
                shs-100k/*.csv          SHS-100K-official-2025 metadata
                      │
     [1] download_shs100k.py            YouTube audio  →  data/audio/
                      │
     [2] align_covers.py                beat-sync chroma  →  data/chroma/
         (chroma → align)               OTI + Smith-Waterman warping paths
                      │                                  →  data/alignments/
     [3] build_manifest.py              quality-filtered tracks + pairs
                      │                                  →  data/manifests/
     [4] train.py                       frozen MERT → 2 encoders → 8+8 latent
         representation learning        tokens → decoder; recon + contrastive
                      │                + swap + cycle + decorrelation losses
                      │                                  →  checkpoints/last.pt
     [5] train_flow.py                  discard decoder, freeze encoders;
         detection                      conditional NSF on p(style | content)
                                        + marginal p(content) + Gaussian baseline
```

Stages 1–3 are one-off preprocessing. Stage 4 is the representation learning.
Stage 5 turns the latents into an anomaly score.

---

## Requirements

**Python packages**

| Package | Used by | Notes |
|---|---|---|
| `torch` | everything in `src/` | 2.x |
| `numpy` | most scripts | 2.x |
| `transformers` | `src/models/mert.py` | loads MERT-v1-330M |
| `librosa` | `align_covers.py` | chroma + beat tracking |
| `numba` | `align_covers.py` | optional, but Smith-Waterman is *much* slower without it |
| `zuko` | `src/models/flow.py` | normalizing flows |
| `tensorboard` | `train.py`, `train_flow.py` | imported via `torch.utils.tensorboard` |

```bash
pip install -r requirements.txt
```

`requirements.txt` carries the version floors and documents the one dependency
conflict that actually bites (numba pins a moving NumPy *ceiling*) plus the two
MERT-specific constraints.

**External binaries**: `ffmpeg` (window decoding, chroma) and `ffprobe`
(duration probing) must be on `PATH`.

**Checkouts** — both are cloned into the repo, not pip-installed:

```bash
git clone https://github.com/second-hand-songs/shs-100k/  shs-100k
git clone https://github.com/yt-dlp/yt-dlp               yt-dlp
```

`scripts/download_shs100k.py` imports the vendored `yt-dlp` directly, so it is
never out of sync with what you tested.

> **Current environment note.** As of 2026-08-18 the active interpreter
> (`/opt/anaconda3`, Python 3.12.2) has `torch` 2.12.1, `numpy` 2.4.5 and
> `transformers` 4.53.2, but is **missing `librosa`, `zuko` and `tensorboard`**,
> and its `numba` 0.60 raises `ImportError: Numba needs NumPy 2.0 or less. Got
> NumPy 2.4.` at import. Stages 2, 4 and 5 will not run until this is fixed —
> `pip install -r requirements.txt` installs what is missing and upgrades numba
> past the NumPy ceiling.

---

## The dataset

`shs-100k/` is **SHS-100K-official-2025**, published by SecondHandSongs. One
headerless CSV per split, five columns:

```
performance_id, work_id, title, artist, youtube_id
8856,8855,1999,"Paul Shaffer & The Party Boys of Rock 'n' Roll",1nuh8t3BQVM
```

Note the column order — *performance* id first, *work* id second — and that
fields are properly CSV-quoted, because titles and artists contain commas.
Watch URLs are not stored; they are rebuilt as
`https://youtube.com/watch?v=<youtube_id>`.

Vocabulary: a **song** (SHS "work") is a composition, i.e. a clique of covers; a
**track** (SHS "performance") is one specific cover. Every track is identified
throughout the codebase by the key `{work_id}_{performance_id}`, which is also
its filename stem in `data/audio/`, `data/chroma/` and the manifests.

`scripts/shs100k_meta.py` is the single place that parses the dataset. It also
corrects two problems in the release:

* **Exact duplicate rows** — 17,154 of train's 116,381. Dropping them is
  required, not cosmetic: two download workers handed the same key would race
  on the same output file.
* **Split contamination.** The upstream README claims works are disjoint across
  subsets. They are not: 40 work ids appeared in both train and test, 27 in
  train and validate, and 3,031 YouTube videos appeared in more than one split.
  `split_tracks()` drops from **train** every track whose work *or* whose own
  video occurs in validate or test — 65 works, 3,032 tracks, 3.1% — leaving zero
  residual overlap and no clique below two versions. Pass `drop_leaked=False` to
  reproduce the contaminated baseline on purpose.

  Validate and test still share 2 works and 76 videos **with each other**. That
  is deliberately unresolved (which of the two should lose them is an evaluation
  decision); the loader prints a warning. Adding `"val": ("test",)` to
  `HELD_OUT_AGAINST` resolves it in favour of test.

Resulting scale:

| split | CSV rows | usable tracks | works | mean clique | largest clique |
|---|---|---|---|---|---|
| train | 116,381 | 96,195 | 1,639 | 58.7 | 1,991 |
| validate | 6,177 | 6,138 | 73 | 84.1 | 898 |
| test | 7,108 | 7,033 | 111 | 63.4 | 595 |

109,366 tracks to fetch; at a measured 3.84 MB average and 76% yield after link
rot, roughly **83,000 files / 320 GB**.

---

## Operating the pipeline

Every script is resumable and safe to interrupt. All of them take
`--data-root` (default `data/`) and `--split {train,val,test,all}`.

### 1. Download audio

```bash
python scripts/download_shs100k.py --split val --limit 20          # smoke test
python scripts/download_shs100k.py --split train --workers 16       # real run
```

Audio-only streams, native codec, no re-encoding, written to
`data/audio/{split}/{key}.{ext}`. Tracks already on disk are skipped, so
re-running resumes. Failures append to `data/logs/download_{split}.csv` so the
real yield can be measured.

The bottleneck is YouTube's per-stream pacing (~2.2 Mbps, roughly playback
rate), **not** your bandwidth — raise `--workers`, don't investigate the
connection. Expect 3–6 days of continuous uptime for the full dataset.

Outcomes are classified in the log so a re-run can target what is worth
retrying:

| status | retryable | meaning |
|---|---|---|
| `ok` | — | downloaded |
| `gone` | no | link rot: unavailable, private, removed, terminated. The expected ~24% |
| `blocked` | no | needs an account: age-restricted, members-only. Permanent here, since the pipeline downloads anonymously by design |
| `throttled` | yes | YouTube rate-limited the session. Slow down and re-run |
| `failed` | yes | anything else — network, extractor hiccup |
| `filtered` | no | longer than `--max-duration` (default 900 s) |

**Re-runs skip the permanent statuses.** They are read back from the log at
startup, so a second pass targets only `throttled` and `failed` instead of
re-confirming that a quarter of the dataset is still dead. Without that, late
runs look like they are doing nothing but hitting unavailable videos — because
that is exactly what they would be doing. `--retry-permanent` forces another
attempt if you think a classification was wrong.

Note that `Sign in to confirm you're not a bot` (throttling) and `Please sign
in` / `Sign in to confirm your age` (a genuinely restricted video) both contain
"sign in" but are opposite diagnoses, so the throttle patterns are matched
first.

**Run anonymously.** If you hit *"Sign in to confirm you're not a bot"* or
*"The page needs to be reloaded"*, that is rate limiting, and cookies are the
wrong fix — they make it worse in three separate ways:

* yt-dlp switches from `_DEFAULT_CLIENTS` (`visionos`, `android_vr`, `web`) to
  `_DEFAULT_AUTHED_CLIENTS` (`tv_downgraded`, `web`); the anonymous set is
  chosen precisely to avoid PO-token and bot-check paths
  (`yt-dlp/yt_dlp/extractor/youtube/_video.py:143`).
* YouTube's ~1-hour rate limit becomes scoped to *the account* rather than the
  session, which caps scale-out no matter how many machines you run
  (`_video.py:4068`).
* A running browser rotates the session cookies out from under the download;
  yt-dlp detects this and warns about it (`_base.py:820`).

The real fix is pacing, and **randomized sleep is not interchangeable with a
lower worker count**. Average request rate is only part of what YouTube reacts
to; burstiness is the rest. With no pre-download sleep, a worker hits the player
API the instant its previous download finishes, and independent workers drift
into sync — so the *peak* rate spikes even when the average looks safe. This is
why 4 workers with no sleep gets bot-checked while more workers with sleep run
clean.

So `--sleep-interval` / `--max-sleep-interval` default to **10–20 s**, matching
yt-dlp's own `-t sleep` preset, which is what YouTube's rate-limit message
recommends. The randomization is the point: it de-synchronizes workers.
`--sleep-requests` (default 1 s) additionally spaces the extraction API calls,
where the bot check lives.

Since the sleep is per worker, throughput is bought back with `--workers`:

| workers | tracks/hr | days for 109,366 |
|---|---|---|
| 4 | 391 | 11.6 |
| 8 | 783 | 5.8 |
| 16 | 1,565 | 2.9 |
| 24 | 2,348 | 1.9 |
| 32 | 3,130 | 1.5 |

Tune `--workers` upward against the `throttled` count and the `tracks/hr`
readout the script prints, keeping the sleep on throughout. Do not trade the
sleep away for fewer workers — that is the combination that gets blocked.
`--player-client` is an escape hatch for forcing specific clients.

#### Before uploading to S3

Mixed containers are expected and need **no conversion**. yt-dlp keeps the
native codec, so you get `.webm` (Opus, 48 kHz) and `.m4a`/`.mp4` (AAC,
44.1 kHz) side by side — and every read path normalizes to 24 kHz mono float32
at decode time anyway (`windows.py`, `align_covers.py`, `build_manifest.py` all
shell out to ffmpeg). Container and source sample rate are erased before
anything in the pipeline sees them, and `existing_audio()` keys on the filename
stem, so the extension never enters an identity.

Upload the originals unchanged. They are the archival copy, and re-acquiring
them is the slow, rate-limited step; transcoding to a uniform 24 kHz mono codec
would save ~3× storage (about $5/month) at the cost of a lossy→lossy generation
and a one-way door.

What is worth catching first is the small stuff that wastes bandwidth or breaks
a preprocessing worker hours in:

```bash
python scripts/check_audio.py --split all --workers 16
```

It reports the codec/sample-rate/channel spread and flags four problems, exiting
non-zero so it can gate an upload script:

| flagged | why it happens | fix |
|---|---|---|
| carries a video stream | `bestaudio/best` falls back to a *combined* stream when a video has no audio-only format — many times larger for the same audio | delete, re-download |
| unreadable | truncated or corrupt write | delete, re-download |
| shorter than the training window | truncated download | delete, re-download |
| leftover `.part` | interrupted download | delete; `existing_audio()` already ignores them, so they would upload as pure waste |

Deleting a flagged file is enough to queue it for re-download — the resume logic
picks up anything missing on the next run.

### 2. Chroma and cover alignment

Covers are not time-aligned — they differ in key, tempo and structure — so
pairing window *k* of cover A with window *k* of cover B would manufacture false
positives. This stage computes the warping paths offline, once.

```bash
python scripts/align_covers.py --stage chroma --split train --workers 2
python scripts/align_covers.py --stage align  --split train --workers 32
```

* `chroma` — beat-synchronous `chroma_cqt` on the harmonic component,
  median-aggregated between beats, L2-normalized → `data/chroma/{split}/{key}.npz`.
  Memory-hungry: each worker can exceed 1 GB, so keep `--workers` low on a small
  machine. `--max-seconds` (default 600) caps HPSS/CQT memory per track.
* `align` — for each pair in a clique, the best of 12 chroma transpositions
  (Optimal Transposition Index) then Smith-Waterman local alignment, stored as
  aligned time arrays plus a normalized confidence score →
  `data/alignments/{split}/{work}_{verA}_{verB}.npz`.

**`--max-pairs-per-song` (default 200) matters.** Cliques reach 1,991 versions,
so the uncapped train split is ~8.9M alignments — about 2,500 core-hours and
8.9M tiny files. The default yields ~327,800 pairs, already far more cover
supervision than Phase 1 consumes. Sampling is seeded by work id, so re-runs
select the same pairs and the stage stays resumable.

Local alignment also cleans the data for free: mislabeled pairs, remixes and
medleys simply fail to align and score low.

### 3. Build manifests

```bash
python scripts/build_manifest.py --split train --min-score 0.2
```

Scans the audio actually on disk, probes durations, cross-references the
alignments, and writes `data/manifests/{split}/`:

* `tracks.jsonl` — every usable track (downloaded, long enough)
* `pairs.jsonl` — every pair whose alignment score clears `--min-score`

`0.2` is the validated threshold (cross-song negative p95; ROC-AUC 0.929,
keeps 81% of true cover pairs on a 50-track val sample — see
`validate_alignment.py`). Tracks that never pair still appear in
`tracks.jsonl` and feed plain reconstruction, which is the correct outcome.

The manifest — not the download — is what defines the training set, so this
script re-applies the split de-contamination to whatever it finds on disk, and
also excludes stale files whose keys are not in the dataset at all.

### 4. Representation learning (Phase 1, online MERT)

```bash
python scripts/train.py --train-split train --val-split val
```

Each step draws `--batch-pairs` aligned cover pairs plus `--batch-tracks` plain
windows, runs frozen MERT-v1-330M on the fly, and optimizes all five objectives
from `CLAUDE.md`: plain reconstruction, MIL-NCE content contrastive, cover-swap
reconstruction, latent cycle-consistency, content–style decorrelation.

Checkpoints go to `checkpoints/last.pt` (every `--checkpoint-every` steps, and
at each epoch end); TensorBoard logs to `runs/<timestamp>/`. Resume with
`--resume checkpoints/last.pt`.

**What Phase 1 is actually for.** Each encoder branch owns a learnable
softmax-weighted sum over MERT's 25 hidden states. Those two 25-dim vectors are
the point of this phase — they are logged every epoch together with their cosine
similarity to the previous epoch:

```
layer weights [content] cosine to previous epoch: 0.999871
```

When that sits at ≈1.0 for two or three consecutive epochs the weights have
converged. Re-run with `--freeze-layer-weights` (or move to Phase-2 caching, not
yet implemented). Do **not** train Phase 1 to convergence of the *losses* — that
is not what it is for.

Throughput knobs: `--mert-micro-batch` controls how many windows go through MERT
per forward (memory), `--num-workers` feeds the ffmpeg window decoders. Each
step spawns roughly `2·batch_pairs + batch_tracks` ffmpeg decodes, so a
CPU-starved machine will idle the GPU.

A local smoke run that fits on a laptop:

```bash
python scripts/train.py --train-split val --val-split val \
    --window-seconds 5 --batch-pairs 2 --batch-tracks 2 \
    --max-steps 5 --device cpu --num-workers 0
```

### 5. Detection flows

```bash
python scripts/train_flow.py --encoder-checkpoint checkpoints/last.pt
```

Loads a Phase-1 checkpoint, freezes it, discards the decoder (scaffolding), and
trains three parameter-disjoint objectives jointly on human music only:

| Component | Models | Role |
|---|---|---|
| `ConditionalStyleFlow` | `log p(style \| content)` | the research object |
| `MarginalFlow` | `log p(content)` | second term of the factorized score |
| `ConditionalGaussian` | diagonal Gaussian | sanity baseline |

Detection score is `log p(style | content) + α · log p(content)`; sweeping α
(`FlowConfig.alpha`) from 0 to 1 shows where the signal actually lives, which is
a direct test of the core hypothesis. Fixed whitening statistics are fitted on
the first `--whiten-windows` windows before optimization. Song-level scores
aggregate per-window scores with a minimum or low percentile.

---

## Configuration

`src/config.py` holds every dimension and hyperparameter as nested dataclasses
(`MertConfig`, `EncoderConfig`, `BottleneckConfig`, `DecoderConfig`,
`FlowConfig`, `LossConfig`, `DataConfig`, `OptimConfig`). Three layers, each
overriding the previous:

1. dataclass defaults
2. `--config path/to/overrides.json` — nested keys only need to name what changes
3. explicit CLI flags

```json
{ "encoder": { "n_layers": 6 },
  "loss":    { "decorrelation": "hsic", "swap_weight": 0.1 },
  "data":    { "window_seconds": 30.0, "n_candidates": 4 } }
```

Checkpoints store the full config, so `config_from_dict` rebuilds the exact
architecture on resume. Tensor shapes are `(B,N,1024)` for MERT and encoder
output, `(B,8,256)` for each of the content and style latents.

---

## Validation and smoke tests

| Command | Checks |
|---|---|
| `python scripts/validate_alignment.py --split val --negatives 200` | Alignment score separation: within-song positives vs. random cross-song negatives, with ROC-AUC, to pick `--min-score` |
| `python scripts/smoke_test_windows.py --n-pairs 8` | Window datasets end to end; aligned pairs must outscore both a time-shifted same-track control and a different-song control |
| `python scripts/smoke_test_model.py` | Every module's shapes, all losses forward+backward, decoder smaller than one encoder branch. `--with-mert` adds real audio through frozen MERT |
| `python scripts/smoke_test_flow.py` | Whitening, flow log-probs, factorized score, invertibility, aggregation, Gaussian baseline |

The time-shifted control in `smoke_test_windows.py` is the sharp one: it fails
if window centers are mapped through the warping path incorrectly.

---

## Layout

```
CLAUDE.md                design rationale and research decisions
README.md                this file
requirements.txt         Python dependencies, with the version traps documented
shs-100k/                dataset metadata checkout (CSVs)
yt-dlp/                  vendored downloader
mert.py                  scratch: standalone MERT exploration, not part of the pipeline
ideas/                   research notes

scripts/
  shs100k_meta.py        the only dataset parser: Track, split_tracks, de-contamination
  download_shs100k.py    [1] audio acquisition
  align_covers.py        [2] chroma + Smith-Waterman alignment
  build_manifest.py      [3] tracks.jsonl / pairs.jsonl
  train.py               [4] representation learning
  train_flow.py          [5] detection flows
  validate_alignment.py  alignment threshold selection
  check_audio.py         pre-upload validation of downloaded audio
  smoke_test_*.py        fast correctness checks

src/
  config.py              all hyperparameters
  losses.py              MIL-NCE, standardized MSE, cycle, HSIC / cross-correlation
  training.py            loss orchestration shared by scripts and smoke tests
  data/
    manifest.py          TrackEntry / PairEntry, jsonl IO
    windows.py           on-the-fly S-second window datasets (ffmpeg seek per item)
  models/
    attention.py         multi-head attention + RoPE
    transformer.py       encoder blocks
    bottleneck.py        cross-attention bottleneck (8 learned latent queries)
    decoder.py           Perceiver-IO-style query decoder
    mert.py              frozen MERT wrapper + learnable LayerMix
    model.py             DisentanglementModel, Standardizer
    flow.py              conditional NSF, marginal flow, pooler, whitener, scoring

data/                    all generated artifacts  (see Version control below)
  audio/{split}/         {work}_{performance}.{ext}
  chroma/{split}/        {key}.npz
  alignments/{split}/    {work}_{verA}_{verB}.npz
  manifests/{split}/     tracks.jsonl, pairs.jsonl
  logs/                  download_{split}.csv
checkpoints/             last.pt
runs/                    TensorBoard logs
```

Windows are never materialized to disk: `src/data/windows.py` seeks into the
source file with ffmpeg and decodes exactly one window per `__getitem__`.

## Version control

`.gitignore` currently contains only `CLAUDE.md`, and **`data/` is tracked** —
170 files under it are in the index and `.git` is already 155 MB. That is
survivable at val-subset scale and catastrophic once a 320 GB download lands, so
before running stage 1 on the real splits:

```bash
cat >> .gitignore <<'IGNORE'
data/
checkpoints/
runs/
__pycache__/
*.pyc
.DS_Store
shs-100k/
IGNORE

git rm -r --cached data                 # stop tracking, keep the files on disk
```

`shs-100k/` and `yt-dlp/` are separate clones with their own `.git`; ignore them
here, or register them as submodules if you want the versions pinned.

---

## Running on AWS

Local machines should be used for smoke tests only — the full dataset and the
GPU training do not fit on a laptop. There is a separate beginner-oriented
runbook covering the AWS stack (S3 + one `c7i` for preprocessing + one `g5.2xlarge`
for training), the account setup order, cost estimates and the traps:

**https://claude.ai/code/artifact/b3c10437-11ad-48dc-8490-cb95b9c31cb0**

The short version: request the GPU quota increase first (new accounts start at
**0** vCPUs for G instances), keep everything in one region, download the
dataset off-AWS, and put audio on the instance's local NVMe rather than reading
it over the network — the dataloader does random ffmpeg seeks.

---

## Status

Implemented and smoke-tested: stages 1–5 above, the full model stack, all five
representation-learning objectives, and the flow-based detection head.

Not yet implemented:

* **Phase-2 feature caching.** `train.py` runs MERT online every step. The
  two-phase plan in `CLAUDE.md` — freeze the layer weights, then cache the two
  weighted-average sequences per track and retrain from those — has no script.
* **Evaluation on generated music.** The MoM (Melody or Machine) benchmark is
  not wired up, so no end-to-end detection numbers exist yet. This also needs
  the shared normalization pipeline described in `CLAUDE.md` (resample to
  24 kHz mono, loudness-normalize, consider matching codecs) so the detector
  cannot shortcut on YouTube compression artifacts instead of the
  style-given-content signal.
* **Val/test overlap resolution** — see the dataset section.
