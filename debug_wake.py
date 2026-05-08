"""Wake-word debugger.

Captures at the mic's native rate, resamples to Vosk's expected rate, and
prints everything Vosk transcribes. If --grammar is set, the decoder is
constrained to the configured WAKE_WORD plus an [unk] sink (same as the
production listener).

Usage:
    python debug_wake.py
    python debug_wake.py --device 5
    python debug_wake.py --grammar
"""

from __future__ import annotations

import argparse
import json
import queue
import sys

import numpy as np
import sounddevice as sd
import vosk

import config
from audio_utils import pick_input_device, resample_int16


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--grammar", action="store_true")
    args = p.parse_args()

    if not config.VOSK_MODEL_DIR.is_dir():
        print(f"Vosk model dir not found: {config.VOSK_MODEL_DIR}", file=sys.stderr)
        sys.exit(1)

    device, native_rate = pick_input_device(args.device)
    target_rate = config.WAKE_WORD_SAMPLERATE
    print(f"Mic device: {device if device is not None else '(default)'}  "
          f"native={native_rate} Hz  vosk={target_rate} Hz")

    print(f"Loading {config.VOSK_MODEL_DIR}...")
    model = vosk.Model(str(config.VOSK_MODEL_DIR))
    if args.grammar:
        rec = vosk.KaldiRecognizer(
            model, target_rate,
            json.dumps([config.WAKE_WORD, "[unk]"]),
        )
        print(f"Grammar mode -- only listening for '{config.WAKE_WORD}'.")
    else:
        rec = vosk.KaldiRecognizer(model, target_rate)
        print("Open mode -- transcribing everything heard.")

    q: queue.Queue[bytes] = queue.Queue()

    def cb(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(status, file=sys.stderr)
        if native_rate != target_rate:
            mono = np.frombuffer(bytes(indata), dtype=np.int16)
            mono = resample_int16(mono, native_rate, target_rate)
            q.put(mono.tobytes())
        else:
            q.put(bytes(indata))

    block = max(1, int(native_rate / 2))  # 500 ms
    print("Speak now. Ctrl-C to stop.\n")
    with sd.RawInputStream(
        samplerate=native_rate,
        blocksize=block,
        device=device,
        dtype="int16",
        channels=1,
        callback=cb,
    ):
        try:
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    text = json.loads(rec.Result()).get("text", "")
                    if text:
                        print(f"FINAL  : {text!r}")
                else:
                    partial = json.loads(rec.PartialResult()).get("partial", "")
                    if partial:
                        print(f"partial: {partial!r}", end="\r")
        except KeyboardInterrupt:
            print("\nbye")


if __name__ == "__main__":
    main()
