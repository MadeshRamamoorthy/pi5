"""Wake-word debugger.

Prints everything Vosk transcribes (no grammar restriction), so you can
see whether the model is even hearing you and how it transcribes "hello
echo". If it consistently transcribes the phrase as something else (e.g.
"hello eco", "low echo"), put that exact transcription into config.WAKE_WORD.

Usage:
    python debug_wake.py
    python debug_wake.py --device 5
    python debug_wake.py --grammar     # use the same tight grammar as the app
"""

from __future__ import annotations

import argparse
import json
import queue
import sys

import sounddevice as sd
import vosk

import config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--samplerate", type=int, default=config.WAKE_WORD_SAMPLERATE)
    p.add_argument("--grammar", action="store_true",
                   help="Constrain decoder to the keyword + [unk] sink "
                        "(same as the app).")
    args = p.parse_args()

    if not config.VOSK_MODEL_DIR.is_dir():
        print(f"Vosk model dir not found: {config.VOSK_MODEL_DIR}", file=sys.stderr)
        print("Download it -- see README §2.8.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {config.VOSK_MODEL_DIR}...")
    model = vosk.Model(str(config.VOSK_MODEL_DIR))
    if args.grammar:
        rec = vosk.KaldiRecognizer(
            model, args.samplerate,
            json.dumps([config.WAKE_WORD, "[unk]"]),
        )
        print(f"Grammar mode -- only listening for '{config.WAKE_WORD}'.")
    else:
        rec = vosk.KaldiRecognizer(model, args.samplerate)
        print("Open mode -- transcribing everything heard.")

    q: queue.Queue[bytes] = queue.Queue()

    def cb(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    print(f"Speak now. Ctrl-C to stop.\nDevice: {args.device or 'default'}\n")
    with sd.RawInputStream(
        samplerate=args.samplerate,
        blocksize=8000,
        device=args.device,
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
