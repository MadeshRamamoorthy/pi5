"""Shared audio helpers — auto-detect the mic's native rate and resample
to whatever Vosk wants (typically 16 kHz)."""

from __future__ import annotations

import os

import numpy as np
import sounddevice as sd


def pick_input_device(explicit: int | None = None) -> tuple[int | None, int]:
    """Return (device_index, native_samplerate).

    Order of preference: explicit arg → SD_DEVICE env var → first
    input-capable device.

    We query WITHOUT the kind="input" filter and validate
    max_input_channels ourselves. sounddevice's strict input-kind check
    rejects duplex USB devices (e.g. the Anker PowerConf conferencing
    speaker) with "Not an input device" even though they expose mic
    channels -- PortAudio flags them as primarily output. Querying
    plain and checking the channel count works for those devices.
    """
    if explicit is None:
        env = os.environ.get("SD_DEVICE")
        if env is not None:
            try:
                explicit = int(env)
            except ValueError:
                pass

    if explicit is None:
        # No explicit pick -- scan for the first device with input
        # channels, preferring USB / PowerConf / Anker over built-ins,
        # and skipping the virtual sysdefault/default/hdmi entries that
        # report 0 input channels.
        best = None
        for idx, dev in enumerate(sd.query_devices()):
            if int(dev.get("max_input_channels", 0)) < 1:
                continue
            name = dev.get("name", "").lower()
            preferred = any(k in name for k in ("usb", "powerconf", "anker", "mic"))
            if best is None or (preferred and not best[1]):
                best = (idx, preferred)
        if best is not None:
            explicit = best[0]

    info = sd.query_devices(explicit)
    if int(info.get("max_input_channels", 0)) < 1:
        raise ValueError(
            f"device {info.get('name', explicit)!r} has no input channels"
        )
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
