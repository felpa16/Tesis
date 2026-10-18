"""Per-epoch detection metrics for train_mert_flow.py --track-metrics.

Scoring MoM the way scripts/evaluate_mom.py does costs a MERT pass over every
window, minutes per evaluation. MERT is frozen and the MERT-only flow consumes
time-pooled hidden states, so the pass is done once: the evaluation songs are
sampled, cut and pooled exactly as evaluate_mom.py samples, cuts and pools
them (same deterministic_sample, window_starts, decode_window and pool_layers),
all 25 layers are stored, and every epoch is then scored from that cache in
seconds. The unfrozen arm's changing layer mix is applied at scoring time.

The human reference is val50 by default, not evaluate_mom.py's test50: these
numbers are looked at during training, and test50 is scored once per run.

Cache layout ({metrics-cache}/):

    meta.json       what was cached: human split, songs per source, K, S, seed
    pooled.npy      (n_windows, 25, 1024) fp16
    windows.jsonl   one {"source", "song", "start"} line per row of pooled.npy
"""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

import evaluate_mom as em
from src.config import MertConfig
from src.data.windows import WindowConfig, decode_window
from src.models import MertExtractor
from src.models.mert_flow import pool_layers

IDENTITY = ("human_split", "per_source", "windows_per_song", "window_seconds", "seed",
            "generators", "model_name")


def song_loader(opener: em.AudioOpener, window: WindowConfig, windows_per_song: int):
    """song -> em.Loaded with its K windows, cut exactly as evaluate_mom.py cuts
    them. A song that cannot be fetched or decoded comes back with .error set."""

    def load(song: em.Song) -> em.Loaded:
        try:
            with opener.open(song) as path:
                duration = song.duration if song.duration is not None else em.probe_duration(path)
                starts = em.window_starts(duration, window.window_seconds, windows_per_song)
                waves = [decode_window(path, start, window) for start in starts]
            return em.Loaded(song, duration, starts, waves)
        except Exception as exc:  # noqa: BLE001 - reported, the song is left out
            return em.Loaded(song, None, [], [], f"{type(exc).__name__}: {exc}")

    return load


def build_eval_cache(
    out_dir: Path,
    *,
    data_root: Path,
    human_split: str,
    generators: list[str],
    per_source: int,
    windows_per_song: int,
    window_seconds: float,
    seed: int,
    bucket: str,
    region: str | None,
    mom_prefix: str,
    mom_root: Path | None,
    mert_config: MertConfig,
    device: torch.device,
    batch: int = 16,
    fetch_workers: int = 8,
) -> None:
    from src.s3 import S3

    meta = {
        "human_split": human_split,
        "per_source": per_source,
        "windows_per_song": windows_per_song,
        "window_seconds": window_seconds,
        "seed": seed,
        "generators": list(generators),
        "model_name": mert_config.model_name,
        "data_root": str(data_root),
    }
    s3 = S3(bucket, region) if mom_root is None else None
    groups = [em.deterministic_sample(
        em.human_songs(data_root, human_split, window_seconds), per_source, seed
    )]
    for generator in generators:
        if mom_root is not None:
            _, songs = em.mom_songs_local(mom_root, generator)
        else:
            _, songs = em.mom_songs_s3(s3, mom_prefix, generator)
        groups.append(em.deterministic_sample(songs, per_source, seed))
    opener = em.AudioOpener(data_root, s3, None, mom_root is not None, None)
    missing = [s for s in groups[0] if not opener.local_path(s).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(groups[0])} {human_split} tracks are not under "
            f"--data-root {data_root}, e.g. {opener.local_path(missing[0])}"
        )
    todo = em.interleave(groups)
    print(f"metrics cache: {len(todo)} songs ({', '.join(f'{len(g)}' for g in groups)} "
          f"for {human_split}, {', '.join(generators)}) -> {out_dir}")

    window = WindowConfig(window_seconds=window_seconds, sample_rate=mert_config.sample_rate)
    load = song_loader(opener, window, windows_per_song)
    mert = MertExtractor(mert_config).to(device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    pooled: list[np.ndarray] = []
    rows: list[dict] = []
    pending: list[tuple[dict, torch.Tensor]] = []
    failures: list[str] = []
    started = time.time()

    def flush(items: list[tuple[dict, torch.Tensor]]) -> None:
        with torch.no_grad():
            out = pool_layers(mert, torch.stack([w for _, w in items]).to(device),
                              mert_config.micro_batch, autocast)
        pooled.append(out.half().cpu().numpy())
        rows.extend(row for row, _ in items)

    for i, loaded in enumerate(em.prefetch(load, todo, fetch_workers)):
        if loaded.error is not None:
            failures.append(f"{loaded.song.source}/{loaded.song.song_id}: {loaded.error}")
            continue
        for start, wave in zip(loaded.starts, loaded.waves):
            pending.append(({"source": loaded.song.source, "song": loaded.song.song_id,
                             "start": start}, wave))
        while len(pending) >= batch:
            flush(pending[:batch])
            pending = pending[batch:]
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(todo)}] songs  {len(rows) / (time.time() - started):.1f} windows/s",
                  flush=True)
    if pending:
        flush(pending)
    del mert
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not rows:
        raise SystemExit("metrics cache: every song failed to load; see the errors above")

    meta["n_windows"] = len(rows)
    meta["failed"] = failures
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "pooled.npy.tmp"
    with open(tmp, "wb") as f:
        np.save(f, np.concatenate(pooled))
    os.replace(tmp, out_dir / "pooled.npy")
    (out_dir / "windows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"metrics cache: {len(rows)} windows in {(time.time() - started) / 60:.1f} min, "
          f"{len(failures)} songs failed")
    for line in failures[:10]:
        print(f"  failed: {line}")


def cache_matches(out_dir: Path, settings: dict) -> bool:
    meta_path = out_dir / "meta.json"
    if not (meta_path.exists() and (out_dir / "pooled.npy").exists()):
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    changed = [k for k in IDENTITY if meta.get(k) != settings[k]]
    if changed:
        raise SystemExit(f"{out_dir} was cached with different {', '.join(changed)}; "
                         "pass another --metrics-cache")
    return True


class EvalCache:
    """The cached evaluation windows, held on the device as fp16 (~0.3 GB)."""

    def __init__(self, root: Path, device: torch.device) -> None:
        self.meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        self.pooled = torch.from_numpy(np.load(root / "pooled.npy")).to(device)
        with open(root / "windows.jsonl", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        self.order = [em.HUMAN, *self.meta["generators"]]

    @torch.no_grad()
    def score(self, model, batch_size: int = 512) -> dict[str, float]:
        """evaluate_mom.py's summary of the current model, flattened for a history line."""
        flow, gaussian = [], []
        for chunk in self.pooled.split(batch_size):
            terms = model.log_probs(chunk.float())
            flow.append(terms["flow"].float().cpu())
            gaussian.append(terms["gaussian"].float().cpu())
        flow_v, gauss_v = torch.cat(flow).tolist(), torch.cat(gaussian).tolist()
        songs: dict[tuple[str, str], dict] = {}
        for row, f, g in zip(self.rows, flow_v, gauss_v):
            record = songs.setdefault((row["source"], row["song"]),
                                      {"source": row["source"], "song": row["song"], "windows": []})
            record["windows"].append({"start": row["start"], "flow": f, "gaussian": g})
        summary, _ = em.summarize(list(songs.values()), self.order, 1.0, "min", 5.0, 0.05, "mert_flow")
        out: dict[str, float | str] = {"metrics/human_split": self.meta["human_split"]}
        for source in self.order[1:]:
            for view in ("flow", "gaussian"):
                out[f"metrics/auroc/{source}/{view}"] = summary["sources"][source]["auroc"][view]
            out[f"metrics/below_threshold/{source}"] = summary["sources"][source]["below_threshold"]
        for view in ("flow", "gaussian"):
            out[f"metrics/auroc/all/{view}"] = summary["all_synthetic"]["auroc"][view]
        out["metrics/below_threshold/all"] = summary["all_synthetic"]["below_threshold"]
        out["metrics/human_median"] = summary["sources"][em.HUMAN]["median"]
        return out
