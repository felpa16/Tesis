# Runbook: learning curve over compositions (25 / 50 / 100 %)

**Question.** Does the representation still get better with more *compositions*
(works), or has it saturated? The answer decides whether the 708 train works
that are not downloaded yet are worth fetching.

**Design.** Three runs that differ only in how many training works they see:

| run | works | what varies |
|---|---|---|
| `lc-w25` | 25 % of the downloaded train works | |
| `lc-w50` | 50 % (contains all of w25) | nested, size-stratified subsets |
| `lc-w100` | all ~931 | |

What is held fixed across the three runs:

* **Pairs per work.** Each work contributes at most 20 aligned cover pairs, so
  "more data" means more compositions, not more pairs of the same huge clique.
* **Budget.** The same number of optimizer steps, with the same LR schedule.
  The smaller subsets simply run more epochs.
* **Evaluation.** `val50` and `test50`: 50 fixed works each, drawn from the
  official held-out splits and never from train.
* **Frozen phase-1 layer mixes** (`phase1_layer_weights.pt`, run #2), and seed 0.

Each run keeps the checkpoint with the best **val50 same-work mAP**, which is
early stopping on held-out content retrieval. That checkpoint is scored on
**test50** once, and the three results are compared pairwise on identical
test windows.

Why online MERT and not the phase-2 cache: the cache for all 931 works
would be ~0.9 TB, and this experiment is what decides how many works the cache
needs to hold. It also saves little, because most of the 1.3 s step is spent in
the encoders, not in MERT. Deferring the cache until the curve is in costs
nothing.

---

## Time and cost (us-east-1, on-demand)

| stage | machine | wall clock | cost |
|---|---|---|---|
| 1. preprocessing | 1 × c7i.16xlarge (64 vCPU, 128 GB) + 500 GB gp3 | ~6 h | ~$18 |
| 2. training | 3 × g5.2xlarge in parallel, one run each | ~18 h | ~$67 |
| 3. evaluation | same g5 boxes | ~15 min | — |
| **total** | | **~1 day** | **~$85** |

Where these numbers come from:

* **Chroma**: ~10 s per 3-minute track per core, measured on the Mac. Budget
  15–20 s per EC2 vCPU, so ~55 k tracks on 56 workers take ~4–5 h.
* **Alignment**: ~0.01 s per pair, so the ~55 k candidate pairs take minutes.
* **Training**: 1.31 s/step at 4+4 windows, measured in phase 1, over ~45 k steps
  (≈ 10 epochs of the 100 % pair set), plus ~5 min per validation.
* Both preprocessing stages are resumable, so spot instances are fine for
  stage 1.

---

## 0. Before you start (Mac)

The new code, the eval designation and the phase-1 weights must be on the
boxes. `phase1_layer_weights.pt` is 21 KB, so commit it:

```bash
cd ~/Tesis
git add scripts/ src/ splits/eval_works.json docs/ phase1_layer_weights.pt
git commit -m "Learning-curve pipeline: kept-pairs alignment, eval designation, subsets"
git push
```

`splits/eval_works.json` is **already generated** (`scripts/designate_eval_works.py`,
seed 0) and is part of the protocol:

| split | works | downloaded tracks | clique size (min / median / max) |
|---|---|---|---|
| `val50` | 50 | 2,681 | 11 / 32 / 450 |
| `test50` | 50 | 2,958 | 11 / 31 / 856 |

The splits are size-matched and disjoint from each other and from train, and
neither shares a video with the other. Do not regenerate them. If you ever
must, `--force` is required, and every earlier result stops being comparable.

Set these in every shell on every box (put them in `~/.bashrc`):

```bash
export BUCKET=<your-bucket>
export AUDIO_S3=s3://$BUCKET/<prefix>      # must contain train/ val/ test/ with the audio
export RUN_S3=s3://$BUCKET/learning-curve  # where this experiment's artifacts go
export DATA=<data root>                    # holds audio/ chroma/ alignments/ manifests/
set -o pipefail                            # a failed download must fail the pipeline
```

Check `AUDIO_S3` before anything else. `aws s3 ls $AUDIO_S3/train/ | head -3`
must list `.webm`/`.m4a` files named like `10154_10161.webm`.

---

## 1. Preprocessing box (CPU)

### 1.1 Launch and environment

c7i.16xlarge, Ubuntu 24.04, a 500 GB gp3 root volume, and an instance profile
with read/write access to the bucket.

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg python3-venv unzip git tmux
curl -s https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o awscli.zip \
  && unzip -q awscli.zip && sudo ./aws/install
git clone https://github.com/felpa16/Tesis.git ~/Tesis && cd ~/Tesis
git clone https://github.com/second-hand-songs/shs-100k/ shs-100k
python3 -m venv ~/venv && echo 'source ~/venv/bin/activate' >> ~/.bashrc && source ~/venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel: src.data imports torch
pip install -r requirements.txt
export DATA=/data && sudo mkdir -p $DATA && sudo chown $USER $DATA   # and add to ~/.bashrc
```

Run everything below inside `tmux`. If you add variables to `~/.bashrc` after
starting tmux, run `tmux kill-server` first. Panes inherit the environment of
the server that started them (`timeline.md`, box B).

### 1.2 Pull the audio

```bash
aws configure set default.s3.max_concurrent_requests 64
for s in train val test; do aws s3 sync "$AUDIO_S3/$s/" "$DATA/audio/$s/" --only-show-errors; done
for s in train val test; do echo "$s $(ls $DATA/audio/$s | wc -l)"; done
```

**Expect** about train 49,809, val 2,308 and test 3,351, i.e. the line counts
of `data/logs/{split}_downloaded_songs.csv`. A large shortfall means a wrong
prefix or an incomplete upload. Fix that before going on.

Optional, but it catches truncated files before a worker trips on them hours
later:

```bash
python scripts/check_audio.py --split all --workers 32 --data-root $DATA
```

Phase-1 chroma, if you archived it on box A, can be restored into
`$DATA/chroma/train/` now. It gets reused and saves ~40 min. Phase-1
alignments are not worth restoring, because re-aligning costs minutes.

### 1.3 Chroma (the long step)

```bash
mkdir -p ~/logs
for s in val test train; do
  python -u scripts/align_covers.py --stage chroma --split $s --workers 56 \
      --data-root $DATA 2>&1 | tee ~/logs/chroma_$s.log
done
```

* val and test run first. They are small, and they smoke-test the environment
  in minutes.
* After ~15 min of train, check the rate with `tail -1 ~/logs/chroma_train.log`:

  ```
  [chroma/train] 4200/51441 (3 failed) 14500 tracks/h, eta 3.3 h
  ```

  A line lands every 100 tracks. If the ETA is far beyond ~5 h, lower
  `--workers` if the box is swapping (`free -g`), or raise it if CPUs are idle
  (`htop`).
* **A progress line is not proof of progress; the output directory is.** Running
  `ls $DATA/chroma/train | wc -l` twice a minute apart answers "is it doing
  anything" independently of how the log is buffered.
* **Expect** a few dozen failures in total (`too few beats`, `too short`). Those
  tracks simply never pair. `UserWarning: Trying to estimate tuning from empty
  frequency set` comes from librosa on near-silent audio; it is harmless, and
  Python prints it once per worker process rather than once per track.

Check: `for s in train val test; do echo "$s $(ls $DATA/chroma/$s | wc -l)"; done`.
Each count should be within ~1 % of its audio count.

### 1.4 Alignment: 20 kept pairs per work

```bash
for s in train val test; do
  python scripts/align_covers.py --stage align --split $s --workers 60 \
      --data-root $DATA --kept-per-song 20 --min-score 0.2 \
      --max-pairs-per-song 200 2>&1 | tee ~/logs/align_$s.log
done
```

Candidates are walked in round-robin order. Every "round" of a work is a
perfect matching, so its 20 pairs are spread over as many distinct recordings
as possible. Each work aligns in parallel rounds until 20 pairs score ≥ 0.2,
or until it has tried 200 candidates. The chosen pairs go to
`alignments/{split}/selection_k20.json`.

**Expect** the last lines of the train log to read roughly:

```
[align/train]   931 works, ~18000 pairs selected; ~890 reached 20, ~20 ran out of candidate pairs, ~20 hit the candidate budget
[align/train]   pairs per work: min 1, median 20, max 20; keep rate ~40% over ~50000 candidates
```

* **~20 "ran out"** is structural: those works have ≤ 6 downloaded tracks,
  so fewer than 20 possible pairs.
* **The keep rate should sit near phase 1's 42 %.** Far below that means chroma
  or alignment is broken. Check a few scores with
  `python -c "import numpy as np,glob; print([float(np.load(f)['score']) for f in glob.glob('$DATA/alignments/train/*.npz')[:20]])"`.
* The stage is resumable. Re-running recomputes nothing and rewrites the same
  selection.

### 1.5 Manifests

```bash
python scripts/build_manifest.py --split train  --data-root $DATA --kept-per-song 20 --workers 32
python scripts/build_manifest.py --split val50  --data-root $DATA --kept-per-song 20 --max-tracks-per-song 40 --workers 32
python scripts/build_manifest.py --split test50 --data-root $DATA --kept-per-song 20 --max-tracks-per-song 40 --workers 32
python scripts/subset_manifest.py --split train --fractions 0.25 0.5 1.0 --data-root $DATA
```

* The eval splits gather each work from the official split it came from
  (`audio/val/…` or `audio/test/…`).
* `--max-tracks-per-song 40` stops the 856-track work from dominating track-level
  metrics. A work's pair tracks are always kept.
* Train keeps every track, because the plain-reconstruction stream benefits
  from every recording.

**Expect**:

| manifest | works | pairs | tracks |
|---|---|---|---|
| `train` / `train_w100` | ~931 | ~18 k | ~49.7 k |
| `train_w50` | ~465 | ~9 k | ~25 k |
| `train_w25` | ~233 | ~4.5 k | ~12 k |
| `val50`, `test50` | 50 each | ~800–1000 each | ≤ 40 per work |

`subset_manifest.py` prints the exact numbers. w25 ⊂ w50 ⊂ w100 by
construction. Each subset holds a proportional share of large and small
cliques, because works are stratified by size in blocks of 4.

Note the training budget now. It is the same number for all three runs:

```bash
STEPS=$(( $(wc -l < $DATA/manifests/train_w100/pairs.jsonl) * 10 / 4 ))
echo "STEPS=$STEPS"   # ≈ 45000 = 10 epochs of the full pair set at 4 pairs/step
```

### 1.6 Upload, verify, terminate

```bash
cd $DATA
tar czf /tmp/alignments_k20.tgz alignments
tar czf /tmp/manifests.tgz manifests
tar czf /tmp/chroma.tgz chroma                 # archive: lets you add pairs later without recomputing
for f in alignments_k20 manifests chroma; do aws s3 cp /tmp/$f.tgz $RUN_S3/preprocessing/$f.tgz; done
aws s3 cp ~/logs $RUN_S3/preprocessing/logs --recursive
aws s3 ls $RUN_S3/preprocessing/
```

Re-download one tarball and count its entries against the local tree before
terminating. That is `timeline.md` blocker 3: a truncated tarball reported
success.

```bash
aws s3 cp $RUN_S3/preprocessing/alignments_k20.tgz /tmp/check.tgz && tar tzf /tmp/check.tgz | grep -c '\.npz$'
find $DATA/alignments -name '*.npz' | wc -l    # must match
```

Then terminate the box.

---

## 2. Training boxes (3 × g5.2xlarge, one per fraction)

The three boxes are identical except for `W`. Launch them together.

### 2.1 Setup (each box)

Deep Learning AMI (PyTorch). The venv lives in `/opt/pytorch`, not in the
system Python (`timeline.md`, blocker 1).

```bash
echo 'source /opt/pytorch/bin/activate' >> ~/.bashrc
cat >> ~/.bashrc <<'EOF'
export DATA=/opt/dlami/nvme/data
export HF_HOME=/opt/dlami/nvme/hf
export BUCKET=<your-bucket>
export AUDIO_S3=s3://$BUCKET/<prefix>
export RUN_S3=s3://$BUCKET/learning-curve
set -o pipefail
EOF
source ~/.bashrc && tmux kill-server 2>/dev/null; mkdir -p $DATA $HF_HOME
sudo apt-get install -y ffmpeg
git clone https://github.com/felpa16/Tesis.git ~/Tesis && cd ~/Tesis
git clone https://github.com/second-hand-songs/shs-100k/ shs-100k
pip install -r requirements.txt

aws configure set default.s3.max_concurrent_requests 64
for s in train val test; do aws s3 sync "$AUDIO_S3/$s/" "$DATA/audio/$s/" --only-show-errors; done
for f in alignments_k20 manifests; do
  aws s3 cp $RUN_S3/preprocessing/$f.tgz /tmp/$f.tgz      # to a file, never piped into tar
  tar tzf /tmp/$f.tgz > /dev/null && tar xzf /tmp/$f.tgz -C $DATA
done
```

Do **not** rebuild manifests here. They reference files by path relative to
`$DATA`, so the ones built in stage 1 are valid as long as the audio and
alignment files are present. Shipping one set also guarantees that all three
runs see byte-identical subsets and eval splits.

`timeline.md` says to build the manifest on the box you train on. That rule
existed because the manifest could reference only the alignments on the box
where it was built. Shipping manifests together with their alignments, and
checking every path, gives the same guarantee. This check proves it:

```bash
python - <<'EOF'
import json, os
root = os.environ["DATA"]
for split in ("train_w100", "val50", "test50"):
    tracks = [json.loads(l) for l in open(f"{root}/manifests/{split}/tracks.jsonl")]
    pairs = [json.loads(l) for l in open(f"{root}/manifests/{split}/pairs.jsonl")]
    missing = [t["audio"] for t in tracks if not os.path.exists(f"{root}/{t['audio']}")]
    missing += [p["alignment"] for p in pairs if not os.path.exists(f"{root}/{p['alignment']}")]
    print(f"{split}: {len(tracks)} tracks, {len(pairs)} pairs, {len(missing)} missing {missing[:3]}")
EOF
```

All three lines must say `0 missing`.

### 2.2 Smoke test (2 minutes; catches environment problems before 18 hours do)

```bash
cd ~/Tesis
python scripts/train.py --data-root $DATA --train-split train_w25 --val-split val50 \
    --layer-weights phase1_layer_weights.pt --freeze-layer-weights \
    --batch-pairs 4 --batch-tracks 4 --num-workers 4 \
    --max-steps 20 --val-every 10 --val-max-batches 5 \
    --checkpoint-dir /tmp/smoke --log-dir /tmp/smoke-runs
```

It must print `layer-mix weights loaded`, `layer-mix weights frozen`,
`trainable parameters: 127.7M` and two validation lines containing
`val/content_work_map`, then finish with `best ...: /tmp/smoke/best.pt`.

### 2.3 Launch (in tmux)

```bash
W=25            # 25 on box 1, 50 on box 2, 100 on box 3
STEPS=<value from 1.5>
mkdir -p checkpoints/lc-w$W
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train.py --data-root $DATA \
    --train-split train_w$W --val-split val50 \
    --layer-weights phase1_layer_weights.pt --freeze-layer-weights \
    --batch-pairs 4 --batch-tracks 4 --num-workers 4 \
    --max-steps $STEPS --val-every 2500 --checkpoint-every 500 \
    --select-metric val/content_work_map \
    --checkpoint-dir checkpoints/lc-w$W --log-dir runs/lc-w$W --seed 0 \
    2>&1 | tee checkpoints/lc-w$W/train.log
```

In a second tmux window, mirror the run to S3 every 30 min. The box can then
die without losing the run.

```bash
while sleep 1800; do aws s3 sync ~/Tesis/checkpoints/lc-w$W $RUN_S3/checkpoints/lc-w$W --only-show-errors; aws s3 sync ~/Tesis/runs/lc-w$W $RUN_S3/runs/lc-w$W --only-show-errors; done
```

**Why these flags:**

* `--batch-pairs 4 --batch-tracks 4` is phase 1's configuration. It is the
  largest that fits the A10G (8+8 went OOM, `timeline.md` blocker 5).
* `--max-steps $STEPS` gives every run the same budget. With it, `train.py`
  takes as many epochs as the subset needs: ~10 for w100, ~20 for w50, ~40
  for w25.
* `--select-metric val/content_work_map` keeps `best.pt` at the point of best
  held-out content retrieval. If w25 overfits, it peaks early and `best.pt`
  keeps the peak, so each run is compared at its best rather than at an
  arbitrary stopping point.
* Same-work pairs in a batch are no longer contrasted as negatives. That was
  the `mil_nce` open item in `timeline.md`. Without the mask, w25 would suffer
  ~4× more false negatives than w100, which would confound the curve.

**What to watch** (TensorBoard over `ssh -L 6006:localhost:6006`, or `train.log`):

* **Every 2,500 steps** comes a validation line. It is appended to
  `checkpoints/lc-w$W/val_metrics.jsonl`, together with the mean training losses
  since the previous validation.
* **Overfitting signature:** `val/contrastive` rising while `train/contrastive`
  keeps falling, and `val/content_work_map` peaking and then declining. This is
  most likely in w25, and it is part of the answer, not a bug.
* **Red flag:** `val/recon` not clearly below the dataset-mean baseline
  (~2.04 at `recon_pool` 16, `timeline.md` run #2). The decoder would not be
  learning.
* **Do not read the per-step lines.** At 4 pairs the contrastive term is a
  4-way classification whose chance level is ln 4 = 1.386, so a single batch
  swings between ~0.05 (all four anchors correct) and ~1.8 (two of four wrong)
  regardless of progress. For a trend, run
  `python scripts/diagnose_training.py checkpoints/lc-w$W/train.log --batch-pairs 4`,
  which blocks the samples and bootstraps the first-vs-last difference.
* **If nothing is moving,** `--overfit-batches 2` retrains on two fixed batches
  with validation off. Every term should collapse toward 0 within a few hundred
  steps; a term that does not is one the model cannot fit even after
  memorizing the data, which points at capacity or optimization rather than at
  the data.
* Validation takes ~5 min per pass (~950 pairs). If that is too slow, add
  `--val-max-batches 120`. The val order is a fixed permutation, so a capped
  pass is still the same subset every time.

**Resume after an interruption:** re-run the identical command with
`--resume checkpoints/lc-w$W/last.pt` added. The step count, LR schedule and
best metric carry over. A validation step can appear twice in
`val_metrics.jsonl` after a resume. That is harmless.

### 2.4 Evaluate on test50 (once, after training)

```bash
python scripts/evaluate_encoder.py --checkpoint checkpoints/lc-w$W/best.pt \
    --split test50 --data-root $DATA 2>&1 | tee checkpoints/lc-w$W/eval_test50.log
python scripts/inspect_phase1.py --checkpoint checkpoints/lc-w$W/best.pt \
    --data-root $DATA --split test50 --batches 50 2>&1 | tee checkpoints/lc-w$W/recon_test50.txt
aws s3 sync checkpoints/lc-w$W $RUN_S3/checkpoints/lc-w$W
aws s3 sync runs/lc-w$W $RUN_S3/runs/lc-w$W
```

* `evaluate_encoder.py` writes `eval_test50.json`: losses, the retrieval
  metrics with 95 % bootstrap intervals over works, and the per-query results.
* `inspect_phase1.py` places test reconstruction between its two trivial
  predictors. Its "fraction of the between-window range" line is the number to
  keep.
* **Touch test50 exactly once per run.** Everything that gets tuned is tuned
  on val50.

Then terminate the box.

---

## 3. Summarize (Mac)

```bash
cd ~/Tesis
for W in 25 50 100; do
  aws s3 sync $RUN_S3/checkpoints/lc-w$W checkpoints/lc-w$W --exclude "*.pt"
done
python scripts/learning_curve.py checkpoints/lc-w25 checkpoints/lc-w50 checkpoints/lc-w100 \
    --split test50 --out checkpoints/learning_curve.md
```

The script prints one row per run:

* training works and pairs
* the step `best.pt` came from
* val numbers at that step, including the train/val contrastive gap
* test50 work mAP and work R@1, each with a 95 % CI over works
* test50 pair R@1, reconstruction and contrastive loss

Below the table come the **paired differences** between consecutive runs:

```
- lc-w50 − lc-w25: Δ test50 work mAP = +0.041 [+0.018, +0.066] paired over 960 queries / 50 works -> real gain
- lc-w100 − lc-w50: Δ test50 work mAP = +0.006 [-0.011, +0.024] paired over 960 queries / 50 works -> within noise
```

(Illustrative numbers.) All three runs are scored on the *same* test windows,
because the sampling is seeded. The difference is therefore bootstrapped query
by query, which is far tighter than comparing two overlapping CIs.

## 4. Reading the result

**Primary metric:** Δ test50 work mAP for w50 → w100.

| w50 → w100 | reading | action |
|---|---|---|
| real gain | the representation is still composition-limited at 931 works | fetch the remaining 708 works (the HF mirror or the scraper), then re-run w100 |
| within noise, and w25 → w50 was a real gain | the curve has flattened by ~900 works | don't block on downloads; move on to phase 2 with what you have |
| within noise everywhere | the metric or the model is the bottleneck, not the data | check the recon and contrastive curves before concluding anything about data |

Also read these:

* **Where `best.pt` came from.** If w25 peaks at a small fraction of `STEPS` while
  w100 peaks near the end, small subsets overfit. That is the overfitting
  question from the data-size discussion, answered directly.
* **Train/val contrastive gap at `best.pt`.** Expect it to shrink as works grow.
  If w100 still shows a large gap, more works (or regularization) would help
  even if the mAP step looks small.
* **Reconstruction** (`recon_test50.txt`) should barely depend on the number of
  works. Recon is per-recording, and even w25 has ~12 k recordings.

**Caveat.** The intervals cover test-set sampling, not training randomness.
If the w50 → w100 call is borderline, repeat w100 with `--seed 1` on one more
box (~$22). Two seeds of the same run show how much of a Δ is seed noise.

Record the table, the paired differences and the decision in `timeline.md`.

---

## Optional: flows on the same subsets

This is not needed for the encoder learning curve. NLLs from different
encoders live in different latent spaces and **cannot be compared**. To see
how the flow's own data needs scale, fix the encoder at `lc-w100/best.pt` and
vary only the flow's training works:

```bash
python scripts/train_flow.py --encoder-checkpoint checkpoints/lc-w100/best.pt \
    --data-root $DATA --train-split train_w$W --val-split val50 \
    --checkpoint-dir checkpoints/flow-w$W --log-dir runs/flow-w$W
```

Compare `val/nll_style` across W, and each run's train–val NLL gap. The gap
is the flow-overfitting signal.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `no pair selection at .../selection_k20.json` | align stage not run (or run without `--kept-per-song`) for that split | run 1.4 for that split |
| `excluded N tracks of designated eval works` when building train | an eval work's audio sits under train (a future designation drawn from train) | expected and correct: eval works never reach training |
| `FileNotFoundError: manifests/...` on a GPU box | `$DATA` unset in that shell | `source ~/.bashrc`; `tmux kill-server` if tmux started before the export |
| `OutOfMemoryError` | batch too large for the A10G | keep 4+4 and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (phase 1 peaked at 18.7 of 22 GB with *trainable* mixes; frozen mixes no longer keep MERT's hidden states for backward, so there is more headroom) |
| chroma prints its header, then nothing for a long time | stdout is block-buffered (~8 KB) behind `tee`, so the every-100-tracks lines pile up invisibly. `align_covers.py` now line-buffers stdout; `python -u` fixes older code | it is almost certainly running: `ls $DATA/chroma/$s \| wc -l` |
| GPU idle, CPU pegged | ffmpeg decoding can't keep up | `--num-workers 6`. If still starved, use g5.4xlarge (16 vCPU) for all three runs, since the runs must match |
