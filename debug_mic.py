"""Live microphone level meter.

Prints an RMS bar and dB level so you can confirm the mic is alive and
loud enough for the wake word. Speak normally; you should see -25 to -10
dBFS bars. If it's silent or stuck near -inf, sounddevice is on the wrong
input device.

USB conferencing mics (e.g. Anker A3301) usually only support 48 kHz, not
16 kHz. By default this script picks the device's native rate; pass --rate
to override.

Usage:
    python debug_mic.py                # default device, native rate
    python debug_mic.py --device 5
    python debug_mic.py --device 5 --rate 48000
    python debug_mic.py --list         # list devices with default rates
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import sounddevice as sd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--rate", type=int, default=None,
                   help="Force samplerate. Default = device's native rate.")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list:
        for i, info in enumerate(sd.query_devices()):
            if info["max_input_channels"] > 0:
                print(f"[{i:>2}] in={info['max_input_channels']} "
                      f"rate={int(info['default_samplerate'])}  "
                      f"{info['name']}")
        print("default device pair:", sd.default.device)
        return

    if args.rate is None:
        info = sd.query_devices(args.device, "input")
        rate = int(info["default_samplerate"])
        print(f"Using device's native rate: {rate} Hz "
              f"({info['name']})")
    else:
        rate = args.rate

    print(f"Recording from device {args.device or 'default'} at {rate} Hz. Ctrl-C to stop.")

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(status, file=sys.stderr)
        samples = indata[:, 0].astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples ** 2)) + 1e-9)
        db = 20 * math.log10(rms)
        bar = "#" * min(60, int((db + 60) * 1.0))
        print(f"\rRMS={rms:7.4f}  {db:6.1f} dBFS  |{bar:<60}|", end="", flush=True)

    with sd.InputStream(
        device=args.device,
        channels=1,
        samplerate=rate,
        dtype="int16",
        blocksize=int(rate / 8),  # ~125ms blocks
        callback=callback,
    ):
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
