"""Decode a window after passing it through a chain of lossy codecs.

Why this exists. Every MoM file is mp3, each generator with its own settings,
and every SHS100K file is YouTube opus or AAC. A model trained to separate the
two can learn the codec instead of the music (experiments.md, section 13).
Two uses:

  * a common chain for both classes ("mp3:64k"): the last codec is then the
    same on both sides, so its settings carry no label
  * a counterfactual: give human songs one generator's own settings (or give
    generator songs YouTube's) and see whether a trained model's score moves.
    The music is unchanged; only the codec history differs.

A chain is stages joined by "+", applied left to right, each
codec:bitrate[:sample_rate[:channels]], e.g. "mp3:64k" or
"opus:160k:48000:2+mp3:64k". Codecs: mp3 (libmp3lame), opus (libopus, which
only takes 48/24/16/12/8 kHz), aac (ffmpeg's native encoder). Defaults: 44.1 kHz
(48 kHz for opus), 2 channels, so a stage mimics a distributed file rather
than re-encoding the 24 kHz mono signal MERT receives, whose artifacts would be
quite different at the same bitrate.

Each stage writes a real file, not a pipe, so the container records the
encoder delay and the decoder removes it. The first stage cuts the window plus
MARGIN_SECONDS on each side, which keeps encoder start-up and the end of the
stream outside the window. The last step is decode_window itself, so the window
is cut and resampled exactly as the uncoded path does it. MERT's input level
does not matter here: its first conv has no bias and is followed by a
per-channel GroupNorm, so a global gain is normalized away.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from src.data.windows import WindowConfig, decode_window

ENCODERS = {"mp3": ("libmp3lame", ".mp3"), "opus": ("libopus", ".opus"), "aac": ("aac", ".m4a")}
OPUS_RATES = (48000, 24000, 16000, 12000, 8000)
MARGIN_SECONDS = 1.0


@dataclass(frozen=True)
class Stage:
    codec: str
    bitrate: str  # ffmpeg -b:a value, e.g. "64k"
    sample_rate: int
    channels: int

    def __str__(self) -> str:
        return f"{self.codec}:{self.bitrate}:{self.sample_rate}:{self.channels}"


Chain = tuple[Stage, ...]


def parse_chain(spec: str | None) -> Chain:
    """"mp3:64k+opus:160k" -> stages; None, "" and "none" -> () (no re-encoding)."""
    if spec is None or spec.strip().lower() in ("", "none"):
        return ()
    stages = []
    for part in spec.split("+"):
        fields = part.strip().split(":")
        if len(fields) < 2 or len(fields) > 4 or fields[0] not in ENCODERS:
            raise SystemExit(f"bad codec stage {part!r}: expected codec:bitrate[:sample_rate[:channels]] "
                             f"with codec one of {', '.join(ENCODERS)}")
        codec, bitrate = fields[0], fields[1]
        rate = int(fields[2]) if len(fields) > 2 else (48000 if codec == "opus" else 44100)
        channels = int(fields[3]) if len(fields) > 3 else 2
        if codec == "opus" and rate not in OPUS_RATES:
            raise SystemExit(f"opus takes {', '.join(map(str, OPUS_RATES))} Hz, not {rate}")
        stages.append(Stage(codec, bitrate, rate, channels))
    return tuple(stages)


def chain_name(chain: Chain) -> str | None:
    """Canonical spelling, stored in cache meta; None when there is no chain."""
    return "+".join(map(str, chain)) if chain else None


def chain_slug(chain: Chain) -> str:
    """For directory names: "mp3:64k:44100:2" -> "mp3-64k-44100-2"."""
    return "_".join(str(s).replace(":", "-") for s in chain)


def check_encoders(chain: Chain) -> None:
    """Fail at start-up, not after hours of failed songs, if ffmpeg lacks an encoder."""
    if not chain:
        return
    listing = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, check=True).stdout
    missing = sorted({ENCODERS[s.codec][0] for s in chain} - set(listing.split()))
    if missing:
        raise SystemExit(f"this ffmpeg has no {', '.join(missing)} encoder; install a build "
                         "with libmp3lame and libopus (e.g. a static ffmpeg build)")


def decode_window_through(
    audio_path: Path, start_seconds: float, config: WindowConfig, chain: Chain
) -> torch.Tensor:
    """decode_window, after re-encoding the window through `chain`."""
    if not chain:
        return decode_window(audio_path, start_seconds, config)
    start = max(start_seconds, 0.0)
    cut = max(start - MARGIN_SECONDS, 0.0)
    with tempfile.TemporaryDirectory() as tmp:
        source = audio_path
        for i, stage in enumerate(chain):
            encoder, suffix = ENCODERS[stage.codec]
            target = Path(tmp) / f"stage{i}{suffix}"
            cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y"]
            if i == 0:
                cmd += ["-ss", f"{cut:.3f}"]
            cmd += ["-i", str(source)]
            if i == 0:
                cmd += ["-t", f"{start - cut + config.window_seconds + MARGIN_SECONDS:.3f}"]
            cmd += ["-vn", "-ac", str(stage.channels), "-ar", str(stage.sample_rate),
                    "-c:a", encoder, "-b:a", stage.bitrate, str(target)]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg {stage}: {result.stderr.decode(errors='replace').strip()}")
            source = target
        return decode_window(source, start - cut, config)
