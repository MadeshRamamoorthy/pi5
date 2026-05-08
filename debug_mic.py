"""Live microphone level meter.

Prints an RMS bar and dB level so you can confirm the mic is alive and
loud enough for the wake word. Speak normally; you should see -25 to -10
dBFS bars. If it's silent or stuck near -inf, sounddevice is on the wrong
input device.

Usage:
    python debug_mic.py                # default input device
    python debug_mic.py --device 5     # specific index from query_devices()
    python debug_mic.py --list         # list devices and exit
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import sounddevice as sd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=None,
                   help="sounddevice index (see --list)")
    p.add_argument("--samplerate", type=int, default=16000)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list:
        print(sd.query_devices())
        print("default:", sd.default.device)
        return

    print(f"Recording from device {args.device or 'default'} "
          f"at {args.samplerate} Hz. Ctrl-C to stop.")

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
        samplerate=args.samplerate,
        dtype="int16",
        blocksize=2048,
        callback=callback,
    ):
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
