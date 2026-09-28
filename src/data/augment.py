"""Style-only waveform augmentation for the contrastive view.

Why style-only. The overfitting measured in timeline.md (2026-09-27) is work
memorisation: with 242 works the contrastive task is solvable by learning which
work each training recording belongs to. Augmentation attacks that directly, by
synthesising extra *performances* of a work the model has already seen.

Every transform here changes timbre and production and leaves pitch and timing
untouched, which is exactly the content/style split the project is built on
(CLAUDE.md): the content encoder should be invariant to all of it. Pitch and
tempo are deliberately *not* augmented — real covers already supply those, and
CLAUDE.md wants key invariance to come from real transpositions rather than from
a resampler.

Everything is FFT or elementwise, so a window costs well under a millisecond and
this runs inside the DataLoader worker. No torchaudio: it is not a dependency,
and an IIR filter in pure Python would be far slower than filtering a spectrum.

The augmented window must never become a reconstruction target. The decoder and
the style branch are fitted only to real recordings, or P(style | content) would
learn that a lowpassed, saturated, reverberant signal is ordinary human style —
which is the distribution the detector later scores against.
"""

from __future__ import annotations

import math
import random

import torch


def _spectral_gain(
    n_freqs: int, sample_rate: int, config, rng: random.Random
) -> torch.Tensor:
    """A smooth random gain curve over rfft bins: tilt, one peak, and a rolloff."""
    freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs)
    # log-frequency axis in [0, 1], so the curve is musical rather than linear
    octaves = torch.log2(freqs.clamp(min=20.0) / 20.0)
    axis = octaves / max(float(octaves[-1]), 1e-6)

    db = torch.zeros(n_freqs)
    db += rng.uniform(-config.tilt_db, config.tilt_db) * (2.0 * axis - 1.0)
    centre = rng.uniform(0.15, 0.85)
    width = rng.uniform(0.08, 0.30)
    db += rng.uniform(-config.peak_db, config.peak_db) * torch.exp(
        -0.5 * ((axis - centre) / width) ** 2
    )

    gain = 10.0 ** (db / 20.0)
    if rng.random() < config.probability:  # codec-like bandwidth limit
        cutoff = rng.uniform(*config.lowpass_hz)
        gain = gain / (1.0 + (freqs / cutoff) ** 8).sqrt()  # 4th-order Butterworth
    return gain


def augment_waveform(
    wave: torch.Tensor, sample_rate: int, config, rng: random.Random
) -> torch.Tensor:
    """One augmented view of a window. Same length, same dtype, same device.

    Each transform fires independently with config.probability, so the identity
    is reachable and the model still sees clean audio sometimes.
    """
    n = wave.shape[-1]
    out = wave.float()
    rms = out.pow(2).mean().sqrt()
    if not torch.isfinite(rms) or rms < 1e-8:  # silent window: nothing to shape
        return wave

    # --- EQ, bandwidth limit and reverb, all in one spectrum multiply ------- #
    spectrum = torch.fft.rfft(out)
    if rng.random() < config.probability:
        spectrum = spectrum * _spectral_gain(spectrum.shape[-1], sample_rate, config, rng)
    if rng.random() < config.probability and config.reverb_seconds > 0:
        # exponentially decaying noise burst = a cheap, plausible small room
        length = max(int(rng.uniform(0.05, config.reverb_seconds) * sample_rate), 16)
        decay = torch.exp(-torch.arange(length, dtype=torch.float32) * 6.0 / length)
        impulse = torch.randn(length, generator=_generator(rng)) * decay
        impulse[0] += 1.0 / max(config.wet, 1e-6)  # keep the dry signal dominant
        impulse = impulse / impulse.abs().sum().clamp(min=1e-6)
        kernel = torch.fft.rfft(impulse, n=n)
        spectrum = spectrum * kernel
    out = torch.fft.irfft(spectrum, n=n)

    # --- dynamics ---------------------------------------------------------- #
    if rng.random() < config.probability and config.saturation > 1.0:
        drive = rng.uniform(1.0, config.saturation)
        peak = out.abs().max().clamp(min=1e-6)
        out = torch.tanh(drive * out / peak) / math.tanh(drive) * peak

    # --- additive noise, set relative to this window's own level ----------- #
    if rng.random() < config.probability:
        snr = rng.uniform(*config.noise_snr_db)
        level = out.pow(2).mean().sqrt() * 10.0 ** (-snr / 20.0)
        out = out + level * torch.randn(n, generator=_generator(rng))

    # --- renormalise to the original loudness, then apply the gain ---------- #
    # Without this, EQ and saturation would leave a level cue that the encoder
    # could read instead of the timbre change.
    new_rms = out.pow(2).mean().sqrt().clamp(min=1e-8)
    out = out * (rms / new_rms)
    if rng.random() < config.probability:
        out = out * 10.0 ** (rng.uniform(-config.gain_db, config.gain_db) / 20.0)

    peak = out.abs().max()
    if peak > 1.0:  # stay in the range a decoded file would occupy
        out = out / peak
    return out.to(wave.dtype)


def _generator(rng: random.Random) -> torch.Generator:
    """A torch Generator seeded from `rng`, so a run is reproducible end to end."""
    generator = torch.Generator()
    generator.manual_seed(rng.randrange(2**63))
    return generator
