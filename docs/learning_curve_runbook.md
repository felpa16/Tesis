# Runbook: do we need more compositions? (two stages)

**Question.** Does the representation still get better with more *compositions*
(works), or has it saturated? The answer decides whether the 708 train works
that are not downloaded yet are worth six days of scraping.

**The design changed on 2026-09-27.** It was three runs varying only the number
of works. `lc-w25` came back overfitting from pass 7 and saturated at step 60 k
(`timeline.md`), and the two fixes for that — a contrastive negative queue and
style-only augmentation — **change the very slope the curve measures**.
Augmentation synthesises extra performances of a work the model already has,
which is the same resource the curve varies, so it should flatten the curve;
the queue stops the objective saturating, which may let the model exploit
diversity it previously ignored, so it should steepen it. Which dominates is not
predictable, and the number that decides the download is the slope **under the
configuration you will ship**.

So the curve is now measured second, after the configuration is settled:

| stage | runs | varies | holds fixed |
|---|---|---|---|
| **A. validate the fixes** | `q-w25`, `qa-w25` | the configuration | data (`train_w25`) |
| **B. the curve** | `w50`, `w100` at stage A's winner | the data | the configuration |

`lc-w25` is stage A's third arm and is already done, so stage A costs two runs
and stage B two more. Every comparison varies one thing.

| run | split | flags beyond the common set |
|---|---|---|
| `lc-w25` (done) | `train_w25` | — (i.e. `--negative-queue 0`) |
| `q-w25` | `train_w25` | `--negative-queue 1000` |
| `qa-w25` | `train_w25` | `--negative-queue 1000 --augment` |
| then `w50`, `w100` | `train_w50`, `train_w100` | the winner's flags |

What is held fixed everywhere:

* **40 kept pairs per work**, so "more data" means more compositions, not more
  pairs of the same huge clique. 40 rather than 20 because the step budget is
  fixed: more pairs mean *fewer passes* over each one, and passes are what drive
  the memorisation (`timeline.md`, 2026-09-27).
* **Budget: 87,372 steps** with the same LR schedule — `lc-w25`'s budget, so it
  stays comparable. Smaller subsets simply run more epochs.
* **Evaluation.** `val50` and `test50`: 50 fixed works each, from the official
  held-out splits, never from train. `val50` holds 1,703 pairs.
* **Frozen phase-1 layer mixes** (`phase1_layer_weights.pt`, run #2), seed 0,
  `--batch-pairs 4 --batch-tracks 4`.

Each run keeps the checkpoint with the best **val50 same-work mAP** — early
stopping on held-out content retrieval — and that checkpoint is scored on
**test50 once**. Runs are compared pairwise on identical test windows.

Why online MERT and not the phase-2 cache: the cache for all 931 works would be
~0.9 TB, and this experiment is what decides how many works it needs to hold.
It also saves little, because most of the 1.3 s step is spent in the encoders,
not in MERT.

---

## Time and cost (us-east-1, on-demand)

| stage | machine | wall clock | cost |
|---|---|---|---|
| 1. preprocessing | 1 × c7i.16xlarge + 500 GB gp3 | ~6 h | ~$18 |
| A. validate the fixes | 2 × g5.2xlarge in parallel | ~19 h | ~$45 |
| B. the curve | 2 × g5.2xlarge in parallel | ~19 h | ~$45 |
| evaluation | the same boxes | ~15 min each | — |
| **total** | | **~2 days** | **~$108** |

Stage 1 is **already done** (K = 40) and its outputs are in
`$RUN_S3/preprocessing/`; keep section 1 for reproduction only.

Where the numbers come from:

* **Training**: 1.31 s/step at 4+4 windows over 87,372 steps ≈ 19 h counting
  ~5 min per validation. Measured, not estimated — `lc-w25` took that.
* **Chroma**: ~10 s per 3-minute track per core on the Mac; budget 15–20 s per
  EC2 vCPU, so ~55 k tracks on 56 workers take ~4–5 h.
* **Alignment**: ~0.01 s per pair; ~93 k candidates at K = 40 take minutes.
* Both preprocessing stages are resumable, so spot instances are fine for
  stage 1. Training mirrors to S3 every 30 min, so spot is survivable there too.

**Run the full budget in stage A, not a cheap short version.** It is tempting to
rank the configurations in 25 k steps for a third of the cost — `lc-w25` was
already at 0.174 mAP by step 22,500 — but the queue's whole purpose is to
prevent the *late* saturation, so a short run would systematically under-measure
it.

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

## 1. Preprocessing box (CPU) — already done, kept for reproduction

This ran at **K = 40** and produced `train_w25/w50/w100`, `val50` and `test50`.
The tarballs are in `$RUN_S3/preprocessing/` (`manifests.tgz`,
`alignments_k40.tgz`, `chroma.tgz`). Skip to stage A and pull them in §A.1
unless you are rebuilding from audio.

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

### 1.4 Alignment: 40 kept pairs per work

```bash
for s in train val test; do
  python -u scripts/align_covers.py --stage align --split $s --workers 60 \
      --data-root $DATA --kept-per-song 40 --min-score 0.2 \
      --max-pairs-per-song 200 2>&1 | tee ~/logs/align_$s.log
done
```

Candidates are walked in round-robin order. Every "round" of a work is a
perfect matching, so its 40 pairs are spread over as many distinct recordings
as possible (~70 % of downloaded recordings land in at least one pair, against
~52 % at K = 20). Each work aligns in parallel rounds until 40 pairs score
≥ 0.2, or until it has tried 200 candidates. The chosen pairs go to
`alignments/{split}/selection_k40.json`.

**40, not 20**, because the step budget is fixed, so pair count sets the number
of passes: w100 makes ~10 passes at K = 40 and would make ~20 at K = 20.
Overfitting in `lc-w25` began at pass 7, so halving the pair count would make it
worse. See `timeline.md` 2026-09-27; a work-aware `clamp(n, 20, 100)` would be
better still but changing it mid-experiment would confound it.

**Expect** the last lines of the train log to read roughly:

```
[align/train]   ~970 works, ~34900 pairs selected; most reached 40, a few ran out of candidate pairs
[align/train]   pairs per work: min 1, median 40, max 40; keep rate ~42% over ~93000 candidates
```

* **A handful of "ran out"** is structural: a work with ≤ 9 downloaded tracks
  has fewer than 40 usable pairs.
* **The keep rate should sit near phase 1's 42 %** — `lc-w25`'s alignments
  measured 41.9 %. Far below that means chroma or alignment is broken. Check a
  few scores with
  `python -c "import numpy as np,glob; print([float(np.load(f)['score']) for f in glob.glob('$DATA/alignments/train/*.npz')[:20]])"`.
* The stage is resumable. Re-running recomputes nothing and rewrites the same
  selection.

### 1.5 Manifests

```bash
python scripts/build_manifest.py --split train  --data-root $DATA --kept-per-song 40 --workers 32
python scripts/build_manifest.py --split val50  --data-root $DATA --kept-per-song 40 --max-tracks-per-song 40 --workers 32
python scripts/build_manifest.py --split test50 --data-root $DATA --kept-per-song 40 --max-tracks-per-song 40 --workers 32
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
| `train` / `train_w100` | ~970 | ~34.9 k | ~52.6 k |
| `train_w50` | ~485 | ~17.5 k | ~26 k |
| `train_w25` | ~242 | ~8.7 k | ~13 k |
| `val50` | 50 | **1,703** | ≤ 40 per work |
| `test50` | 50 | ~1,700 | ≤ 40 per work |

`subset_manifest.py` prints the exact numbers. w25 ⊂ w50 ⊂ w100 by
construction. Each subset holds a proportional share of large and small
cliques, because works are stratified by size in blocks of 4.

The budget is the same number for **every** run in both stages:

```bash
STEPS=$(( $(wc -l < $DATA/manifests/train_w100/pairs.jsonl) * 10 / 4 ))
echo "STEPS=$STEPS"   # 87372 = 10 passes over the full pair set at 4 pairs/step
```

`lc-w25` used 87,372, so anything you want to compare against it must too.

### 1.6 Upload, verify, terminate

```bash
cd $DATA
tar czf /tmp/alignments_k40.tgz alignments
tar czf /tmp/manifests.tgz manifests
tar czf /tmp/chroma.tgz chroma                 # archive: lets you add pairs later without recomputing
for f in alignments_k40 manifests chroma; do aws s3 cp /tmp/$f.tgz $RUN_S3/preprocessing/$f.tgz; done
aws s3 cp ~/logs $RUN_S3/preprocessing/logs --recursive
aws s3 ls $RUN_S3/preprocessing/
```

Re-download one tarball and count its entries against the local tree before
terminating. That is `timeline.md` blocker 3: a truncated tarball reported
success.

```bash
aws s3 cp $RUN_S3/preprocessing/alignments_k40.tgz /tmp/check.tgz && tar tzf /tmp/check.tgz | grep -c '\.npz$'
find $DATA/alignments -name '*.npz' | wc -l    # must match
```

Then terminate the box.

---

## 2. Training boxes (shared setup, both stages)

Two boxes per stage, identical except for the one flag or split that varies.
Launch them together.

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
for f in alignments_k40 manifests; do
  aws s3 cp $RUN_S3/preprocessing/$f.tgz /tmp/$f.tgz      # to a file, never piped into tar
  tar tzf /tmp/$f.tgz > /dev/null && tar xzf /tmp/$f.tgz -C $DATA
done
```

Do **not** rebuild manifests here. They reference files by path relative to
`$DATA`, so the ones built in stage 1 are valid as long as the audio and
alignment files are present. Shipping one set also guarantees that every run in
both stages sees byte-identical subsets and eval splits.

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

Everything shared by every run in both stages:

```bash
STEPS=87372     # from 1.5; lc-w25 used this, so comparisons need it too
COMMON="--data-root $DATA --val-split val50 \
  --layer-weights phase1_layer_weights.pt --freeze-layer-weights \
  --batch-pairs 4 --batch-tracks 4 --num-workers 4 \
  --max-steps $STEPS --val-every 2500 --checkpoint-every 500 \
  --select-metric val/content_work_map --seed 0"

launch () {   # launch <run-name> <train-split> <extra flags...>
  RUN=$1; SPLIT=$2; shift 2
  mkdir -p checkpoints/$RUN
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/train.py $COMMON --train-split $SPLIT "$@" \
      --checkpoint-dir checkpoints/$RUN --log-dir runs/$RUN \
      2>&1 | tee checkpoints/$RUN/train.log
}
```

**Stage A — validate the fixes** (data fixed at `train_w25`):

```bash
launch q-w25  train_w25 --negative-queue 1000              # box 1
launch qa-w25 train_w25 --negative-queue 1000 --augment    # box 2
```

`lc-w25` is the third arm and is already done. **It is no longer the default
configuration**: `--negative-queue` now defaults to 1000, so reproducing or
resuming `lc-w25` requires an explicit `--negative-queue 0`.

**Stage B — the curve**, once stage A has picked a configuration. Substitute its
flags for `<winner>`:

```bash
launch w50  train_w50  <winner>      # box 1
launch w100 train_w100 <winner>      # box 2
```

Stage A's winning w25 run *is* the curve's 25 % point, so do not re-run it.

In a second tmux window, mirror to S3 every 30 min. The box can then die
without losing the run.

```bash
while sleep 1800; do
  aws s3 sync ~/Tesis/checkpoints/$RUN $RUN_S3/checkpoints/$RUN --only-show-errors
  aws s3 sync ~/Tesis/runs/$RUN $RUN_S3/runs/$RUN --only-show-errors
done
```

**Why these flags:**

* `--batch-pairs 4 --batch-tracks 4` is phase 1's configuration, the largest
  that fits the A10G (8+8 went OOM, `timeline.md` blocker 5). The queue is the
  way to add negatives *without* adding windows, which is why it exists.
* `--max-steps $STEPS` gives every run the same budget. `train.py` then takes as
  many epochs as the subset needs: ~10 for w100, ~20 for w50, ~40 for w25.
* `--negative-queue 1000` gives each anchor 3 in-batch negatives plus up to
  1,000 queued ones. Queue entries of the anchor's own work are masked out, the
  same as in-batch ones. Validation does **not** use the queue, so
  `val/contrastive` stays comparable to `lc-w25`'s history.
* `--augment` applies style-only augmentation (EQ, bandwidth, reverb,
  saturation, noise, gain) to each pair's A-side window. Augmented windows are
  excluded from the reconstruction terms and from the standardizer, because
  fitting the decoder and the style branch to them would teach
  P(style | content) that lowpassed, saturated audio is ordinary human style.
  Pitch and tempo are not augmented — real covers supply those. Validate the
  transforms on real audio first:
  `python scripts/check_augment.py --data-root $DATA --split val --n 12`
  (reverb is the one that disturbs content most; turn it down in
  `AugmentConfig` if it looks worse than the ~76 % measured on synthetic audio).
* `--select-metric val/content_work_map` keeps `best.pt` at the point of best
  held-out content retrieval, so each run is compared at its best rather than at
  an arbitrary stopping point. `lc-w25` peaked at step 67,500 of 87,372.
* Same-work pairs in a batch are not contrasted as negatives (the `mil_nce`
  open item). Without the mask, w25 would suffer ~4× more false negatives than
  w100, which would confound stage B.

**What to watch** (TensorBoard over `ssh -L 6006:localhost:6006`, or `train.log`):

* **Every 2,500 steps** comes a validation line, appended to
  `checkpoints/$RUN/val_metrics.jsonl` with the mean training losses since the
  previous validation.
* **The train/val contrastive gap is the number that matters in stage A.**
  `lc-w25` ended at +0.618 (train 0.329, val 0.947), having bottomed out at step
  15,000. A much smaller gap means the fix reduced work memorisation — and that
  is also a direct prediction that stage B's curve will be flatter. A gap near
  +0.6 means memorisation is untouched and works are still the constraint.
* **Where `best.pt` lands.** Moving from step 67,500 toward the end of the run
  means the saturation is fixed.
* **`train/contrastive` is not comparable across stage A.** With a queue it is a
  ~1,000-way loss instead of 4-way, so its absolute value jumps. Judge on
  `val/content_work_map` and the gap trend.
* **Red flag:** `val/recon` not clearly below the dataset-mean baseline
  (~2.02 at `recon_pool` 16). `lc-w25` reached 1.818, which is 71 % of the way
  to a per-window constant — still short of it, and the known `recon_pool`
  limitation rather than a new fault.
* **Do not read the per-step lines.** At 4 pairs the contrastive term is a
  4-way classification whose chance level is ln 4 = 1.386, so a single batch
  swings between ~0.05 (all four anchors correct) and ~1.8 (two of four wrong)
  regardless of progress. For a trend, run
  `python scripts/diagnose_training.py checkpoints/$RUN/train.log --batch-pairs 4`,
  which blocks the samples and bootstraps the first-vs-last difference.
* **If nothing is moving,** `--overfit-batches 2` retrains on two fixed batches
  with validation off. Every term should collapse toward 0 within a few hundred
  steps; a term that does not is one the model cannot fit even after memorising
  the data, which points at capacity or optimisation rather than at the data.
* Validation is 426 batches (1,703 val50 pairs), ~5 min. `--val-max-batches 120`
  caps it; the val order is a fixed permutation, so a capped pass is the same
  subset every time. The retrieval metrics pool the **whole** pass, so a cap
  makes the task easier and must be held constant across anything you compare.

**Resume after an interruption:** re-run the identical command with
`--resume checkpoints/$RUN/last.pt` added. The step count, LR schedule and best
metric carry over; the negative queue does not (its buffers are deliberately not
checkpointed) and refills over the first ~250 steps. A validation step can
appear twice in `val_metrics.jsonl` after a resume. That is harmless.

### 2.4 Evaluate on test50 (once, after training)

```bash
python scripts/evaluate_encoder.py --checkpoint checkpoints/$RUN/best.pt \
    --split test50 --data-root $DATA 2>&1 | tee checkpoints/$RUN/eval_test50.log
python scripts/inspect_phase1.py --checkpoint checkpoints/$RUN/best.pt \
    --data-root $DATA --split test50 --batches 50 2>&1 | tee checkpoints/$RUN/recon_test50.txt
aws s3 sync checkpoints/$RUN $RUN_S3/checkpoints/$RUN
aws s3 sync runs/$RUN $RUN_S3/runs/$RUN
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

`learning_curve.py` takes any list of run directories, so it serves both stages.

```bash
cd ~/Tesis
for RUN in lc-w25 q-w25 qa-w25 w50 w100; do
  aws s3 sync $RUN_S3/checkpoints/$RUN checkpoints/$RUN --exclude "*.pt" 2>/dev/null
done

# stage A: same data, different configuration
python scripts/learning_curve.py checkpoints/lc-w25 checkpoints/q-w25 checkpoints/qa-w25 \
    --split test50 --out checkpoints/stage_a.md

# stage B: same configuration, more works (put the winning w25 run first)
python scripts/learning_curve.py checkpoints/<winner>-w25 checkpoints/w50 checkpoints/w100 \
    --split test50 --out checkpoints/learning_curve.md
```

One row per run: training works and pairs, the step `best.pt` came from, the
val numbers there including the train/val contrastive gap, and test50 work mAP,
work R@1, pair R@1, recon and contrastive — the retrieval numbers with 95 %
bootstrap intervals over works.

Below the table come the **paired differences** between consecutive runs:

```
- w50 − qa-w25: Δ test50 work mAP = +0.041 [+0.018, +0.066] paired over 1703 queries / 50 works -> real gain
- w100 − w50:   Δ test50 work mAP = +0.006 [-0.011, +0.024] paired over 1703 queries / 50 works -> within noise
```

(Illustrative numbers.) Every run is scored on the *same* test windows, because
the sampling is seeded, so the difference is bootstrapped query by query — far
tighter than comparing two overlapping intervals. The same machinery reads
stage A, where the paired difference is the cleanest statement of what the
queue and the augmentation are worth.

## 4. Reading the result

### Stage A — which configuration to carry into stage B

| reading | conclusion |
|---|---|
| `q-w25` > `lc-w25` on test50 work mAP, and `best.pt` moves later in the run | the queue fixed the saturation; keep it |
| `qa-w25` > `q-w25`, **and** the train/val contrastive gap shrinks | augmentation is genuinely reducing work memorisation; keep both, and expect stage B's curve to be flatter |
| `qa-w25` ≈ `q-w25` but the gap is unchanged | augmentation is cosmetic here; drop it rather than carry an unexplained factor into the curve |
| `qa-w25` < `q-w25` | the transforms are destroying content — check `scripts/check_augment.py` on real audio before blaming the idea |

The gap is the load-bearing number, not just the mAP: it is the direct readout
of the mechanism (`lc-w25` ended at **+0.618**), and it predicts stage B's slope
before stage B is run.

### Stage B — the download decision

**Primary metric:** Δ test50 work mAP for w50 → w100.

| w50 → w100 | reading | action |
|---|---|---|
| real gain | still composition-limited at ~970 works | fetch the remaining 708 (the gated HF mirror `Yougen/shs100k_dataset` is 306 GB and one download; the scraper is ~6 days), then re-run w100 |
| within noise, and w25 → w50 was a real gain | the curve has flattened by ~970 works | don't block on downloads; move to phase 2 with what you have |
| within noise everywhere | the metric or the model is the bottleneck, not the data | check recon and the contrastive curves before concluding anything about data |

**Extrapolate rather than eyeball.** Three points give a slope per doubling of
works. Project it to 1,639 (all de-contaminated train works, ~0.75 further
doublings) and to ~10 k (SHS100K-v2 scale) before deciding. 931 works is small
for a contrastive problem — Da-TACOS is ~1,000, SHS100K-v2 ~10 k — so a flat
w50 → w100 step is evidence about *this* configuration's appetite, not proof
that data has stopped mattering.

**State the conclusion conditionally.** The curve is measured under stage A's
winning configuration, and augmentation is a partial substitute for works. The
honest form is "at the configuration we ship, works saturate at N", not "works
saturate at N".

Also read these:

* **Where `best.pt` came from.** If w25 peaks early while w100 peaks near the
  end, small subsets overfit — the overfitting question answered directly.
* **Train/val contrastive gap at `best.pt`.** Expect it to shrink as works grow.
  A large gap at w100 means more works or more regularisation would still help,
  even if the mAP step looks small.
* **Reconstruction** (`recon_test50.txt`) should barely depend on the number of
  works. Recon is per-recording, and even w25 has ~13 k recordings. `lc-w25`
  confirmed this: `val/recon` fell monotonically with no overfitting at all.

**Caveat.** The intervals cover test-set sampling, not training randomness. If
the w50 → w100 call is borderline, repeat w100 with `--seed 1` on one more box
(~$22). Two seeds of the same run show how much of a Δ is seed noise.

Record the tables, the paired differences and the decision in `timeline.md`.

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
| `no pair selection at .../selection_k40.json` | align stage not run (or run without `--kept-per-song 40`) for that split | run 1.4 for that split |
| `excluded N tracks of designated eval works` when building train | an eval work's audio sits under train (a future designation drawn from train) | expected and correct: eval works never reach training |
| `FileNotFoundError: manifests/...` on a GPU box | `$DATA` unset in that shell | `source ~/.bashrc`; `tmux kill-server` if tmux started before the export |
| `OutOfMemoryError` | batch too large for the A10G | keep 4+4 and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (phase 1 peaked at 18.7 of 22 GB with *trainable* mixes; frozen mixes no longer keep MERT's hidden states for backward, so there is more headroom) |
| chroma prints its header, then nothing for a long time | stdout is block-buffered (~8 KB) behind `tee`, so the every-100-tracks lines pile up invisibly. `align_covers.py` now line-buffers stdout; `python -u` fixes older code | it is almost certainly running: `ls $DATA/chroma/$s \| wc -l` |
| a run you meant to match `lc-w25` scores differently | `--negative-queue` defaults to **1000**, so `lc-w25`'s configuration is no longer the default | pass `--negative-queue 0` explicitly to reproduce or resume it; `run_info.json` records the value for every run |
| `train/contrastive` jumps by several nats when you turn the queue on | expected: it is a ~1000-way loss now, not 4-way | compare runs on `val/content_work_map` and the train/val gap, never on the raw train loss |
| `qa-*` scores below `q-*` | the augmentation may be changing content, not style | `python scripts/check_augment.py --data-root $DATA --split val --n 12`; anything far below the clean self-score (reverb is the usual culprit) should be turned down in `AugmentConfig` |
| GPU idle, CPU pegged | ffmpeg decoding can't keep up | `--num-workers 6`. If still starved, use g5.4xlarge (16 vCPU) for all three runs, since the runs must match |
