"""Shared audio helpers — auto-detect the mic's native rate and resample
to whatever Vosk wants (typically 16 kHz)."""

from __future__ import annotations

import os

import numpy as np
import sounddevice as sd


def pick_input_device(explicit: int | None = None) -> tuple[int | None, int]:
    """Return (device_index, native_samplerate).

    Order of preference: explicit arg → SD_DEVICE env var → sounddevice default.
    """
    if explicit is None:
        env = os.environ.get("SD_DEVICE")
        if env is not None:
            try:
                explicit = int(env)
            except ValueError:
                pass
    info = sd.query_devices(explicit, "input")
    rate = int(info["default_samplerate"])
    return explicit, rate


def resample_int16(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Cheap linear-interp resampler for int16 mono audio. Adequate for ASR."""
    if src_rate == dst_rate:
        return samples
    n_src = len(samples)
    if n_src == 0:
        return samples
    n_dst = max(1, int(round(n_src * dst_rate / src_rate)))
    x_src = np.linspace(0.0, 1.0, n_src, endpoint=False)
    x_dst = np.linspace(0.0, 1.0, n_dst, endpoint=False)
    out = np.interp(x_dst, x_src, samples.astype(np.float32))
    return out.astype(np.int16)
