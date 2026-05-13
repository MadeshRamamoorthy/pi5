# Speech recognition on the Hailo NPU

**Status: production**. Hailo ships pre-compiled Whisper encoder +
decoder HEFs and a Python pipeline that runs the whole STT path on
the NPU.

Variants the `hailo-apps` CLI auto-downloads today:

| Variant | Approx latency on Hailo-10H | Accuracy | Hardware support |
|---------|-----------------------------|----------|------------------|
| **Whisper-Tiny**  | ~150-300 ms | basic | Hailo-8 / Hailo-8L / Hailo-10H |
| **Whisper-Base**  | ~250-500 ms (default) | good | Hailo-8 / Hailo-8L / Hailo-10H |
| **Whisper-Tiny.en** | ~150-300 ms | better English than tiny | Hailo-10H only |

**Whisper-Small** has a Model Explorer page on hailo.ai but is **not
in the hailo-apps CLI** (`--variant small` errors out with
`invalid choice: 'small'`). If you need it, watch the community
thread linked at the bottom -- people have built it manually with the
Dataflow Compiler and shared HEFs. The wrapper in `asr.py` will
happily load it once the encoder/decoder HEFs and matching `.npy`
files are on disk; just set `HAILO_WHISPER_MODEL=small` and point
the env vars at the right files.

Hailo Model Explorer pages:

- <https://hailo.ai/products/hailo-software/model-explorer/generative-ai/whisper-tiny/>
- <https://hailo.ai/products/hailo-software/model-explorer/generative-ai/whisper-base/>
- <https://hailo.ai/products/hailo-software/model-explorer/generative-ai/whisper-small/> (no HEF in CLI yet)

For best out-of-the-box accuracy on the Pi 5 + AI HAT 2+, use
**`tiny.en`** (Hailo-10H only). It's the English-specialised tiny
model, faster than base and noticeably more accurate on English
speech than the multilingual tiny.

vs. our other backends:

| Backend | Latency on a 5 s utterance | Cost |
|---------|----------------------------|------|
| **Hailo Whisper-Base** | **~250-500 ms** | $0 |
| OpenAI Whisper API     | ~1-2 s        | ~$0.0002/min |
| faster-whisper tiny.en (Pi 5 CPU) | ~3-4 s | $0 |

## Two integration paths

HailoRT shipped a native Whisper API in 5.2.0. The kiosk supports
both paths and auto-selects:

| Path | When | What it needs |
|------|------|---------------|
| **Native `Speech2Text`** (recommended) | HailoRT ≥ 5.2.0 | One **combined** HEF (encoder + decoder packed together). No `.npy` files, no `add_embed`, no patches. |
| Legacy `hailo-apps.WhisperPipeline` | HailoRT 5.0–5.1 *or* if you only have **separate** encoder + decoder HEFs | Two HEF files + `npy_dir` of decoder tokenization assets + the `add_embed` flag set correctly for your chip. |

The native path is dramatically simpler. If your HailoRT is recent
enough, prefer it.

```bash
hailortcli -v          # must show 5.2.0 or newer
```

To use the native path: set `HAILO_WHISPER_HEF` in `.env` (or
`config.py`) to a combined HEF, and the kiosk's auto-selector picks
`hailo-whisper-native` automatically.

```bash
# .env
HAILO_WHISPER_HEF=/usr/local/hailo/resources/models/hailo10h/Whisper-Small.hef
```

## How the kiosk picks an STT backend

`config.CHAT_ASR_BACKEND = "auto"` (the default) goes through this
precedence:

1. **`hailo-whisper-native`** (preferred) — chosen when
   `HAILO_WHISPER_HEF` points at an existing combined HEF. Uses
   HailoRT 5.2+'s built-in `Speech2Text` API; no application-level
   pipeline code.
2. **`hailo-whisper`** (legacy) — chosen when the separate encoder
   HEF, decoder HEF, AND the `npy_dir` are all on disk and the
   encoder/decoder paths differ. Uses `hailo-apps`'s
   `WhisperPipeline`. The auto-selector deliberately skips this path
   when encoder and decoder paths point at the same file (combined
   HEF) since the legacy code assumes split files.
3. **`openai`** — chosen when `OPENAI_API_KEY` is set.
4. **`faster-whisper`** — pure CPU fallback for fully offline boxes.

Force a specific backend: `CHAT_ASR_BACKEND = "hailo-whisper-native"`
(or `"hailo-whisper"` / `"openai"` / `"faster-whisper"` / `"vosk"`).

## Setup on the Pi 5 + AI HAT 2+ (Hailo-10H)

```bash
# Prereqs from Hailo's README:
sudo apt install -y ffmpeg libportaudio2

# Activate the project venv (PEP 668 means pip install must run inside it
# on Trixie):
cd ~/Documents/code/pi5
source .venv/bin/activate

# Clone hailo-apps and install with the speech-recognition extras.
# It's NOT on PyPI -- pip install 'hailo-apps[speech-rec]' from the
# index will fail with "No matching distribution found".
cd ~/Documents/code
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
pip install -e '.[speech-rec]'

# hailo-apps downloads HEFs into /usr/local/hailo/resources/, which
# isn't writable by your user by default. Fix that once:
sudo mkdir -p /usr/local/hailo/resources/models/hailo10h
sudo chown -R $USER:$USER /usr/local/hailo

# Trigger Hailo's downloader. The CLI smoke-test may fail to record
# audio (USB mics often refuse 16 kHz natively) -- that's fine, the
# download runs BEFORE recording so the files are already on disk
# by the time it errors.
python -m hailo_apps.python.standalone_apps.speech_recognition.speech_recognition \
    --arch hailo10h --variant base --duration 6
# Expected end-state files (note Hailo's "-10s" / "-10s-out-seq-64"
# suffixes baked into the names):
#   /usr/local/hailo/resources/models/hailo10h/base-whisper-encoder-10s.hef
#   /usr/local/hailo/resources/models/hailo10h/base-whisper-decoder-10s-out-seq-64.hef
#   /usr/local/hailo/resources/npy/*.npy                 (token embeddings)
```

**No symlinks needed.** `config.py` checks `pi5/models/` first (for
users who prefer a project-local copy) and falls back to
`/usr/local/hailo/resources/...` automatically. Override either via
env var (`HAILO_WHISPER_ENCODER_HEF`, `HAILO_WHISPER_DECODER_HEF`,
`HAILO_WHISPER_NPY_DIR`) for a custom layout.

## Switching variants

```bash
# 1. Edit config.py (or .env)
#    HAILO_WHISPER_MODEL = "tiny.en"   # or "tiny" / "base"

# 2. Trigger hailo-apps to download the matching HEFs + npy assets.
cd ~/Documents/code/hailo-apps
source ~/Documents/code/pi5/.venv/bin/activate
python -m hailo_apps.python.standalone_apps.speech_recognition.speech_recognition \
    --arch hailo10h --variant tiny.en --duration 6
# (CLI will fail at the recording step on USB mics that refuse 16 kHz;
#  the download phase has already completed, which is what we need.)

# 3. Verify the files arrived
ls -lh /usr/local/hailo/resources/models/hailo10h/ | grep whisper

# 4. Restart the kiosk
cd ~/Documents/code/pi5
./stop.sh && ./start.sh
```

Per-variant `.npy` files live alongside in
`/usr/local/hailo/resources/npy/`
(`token_embedding_weight_<variant>.npy`, etc.) and are downloaded by
the same CLI run. No symlinks needed.

### Trying Whisper-Small (not in hailo-apps CLI)

If you have HEFs from elsewhere (e.g. someone in the community
thread compiled them), the kiosk supports the variant -- it just
doesn't know how to download them.

```bash
# Put the files anywhere readable, then point env vars at them:
# .env
HAILO_WHISPER_MODEL=small
HAILO_WHISPER_ENCODER_HEF=/path/to/small-whisper-encoder-10s.hef
HAILO_WHISPER_DECODER_HEF=/path/to/small-whisper-decoder-10s-out-seq-64.hef
HAILO_WHISPER_NPY_DIR=/path/to/dir/with/small-variant-npy/files

./stop.sh && ./start.sh
```

## Restart the kiosk

```bash
./stop.sh && ./start.sh
```

Verify in `/tmp/echo-backend.log`:
```
[asr] auto-selected backend: hailo-whisper
[chat-voice] using hailo-whisper
... first chat question ...
[asr] loading Hailo Whisper base (encoder=models/whisper-base-encoder.hef,
        decoder=models/whisper-base-decoder.hef,
        npy_dir=models/whisper-base-assets, add_embed=False)
```

## How the wrapper drives the pipeline

`asr.py:HailoWhisperASR` mirrors the reference loop from
`hailo-apps/python/standalone_apps/speech_recognition`:

```python
pipeline = WhisperPipeline(encoder_path, decoder_path,
                           variant="base", npy_dir=..., add_embed=False)
mels = preprocess_audio(audio_float32, chunk_length=pipeline.get_chunk_length())
for mel in mels:
    pipeline.send_data(mel)
    time.sleep(0.1)
    text = pipeline.get_transcription()
    results.append(text)
return clean_transcription(" ".join(results))
```

A `threading.Lock` serialises `transcribe()` calls because the
pipeline keeps internal state across `send_data` / `get_transcription`
pairs.

## `add_embed` (Hailo-8/8L vs Hailo-10H)

Per `hailo-apps/speech_recognition.py`, `add_embed=True` for the
Hailo-8 and Hailo-8L (host CPU runs the token-embedding matmul) and
`add_embed=False` for Hailo-10H (embedding runs on the chip). The
default in `config.py` is `False` — Pi 5 + AI HAT 2+. Flip it for
older hardware:

```python
# config.py
HAILO_WHISPER_ADD_EMBED = True
```

## Chip-contention reality check

The Hailo-10H currently hosts:

- SCRFD (face detect) + ArcFace (embed) — every camera frame while
  ACTIVE
- Hailo-Whisper encoder + decoder — during chat exchanges
- Optionally Hailo-Ollama (`qwen3:1.7b`) — only when
  `PRELOAD_OLLAMA=1` or `OPENAI_API_KEY` is unset

### How they share the chip

Both pipelines create their HailoRT VDevice with
`scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN` and
`group_id = "SHARED"`:

- hailo-apps's `whisper_pipeline.py` does this in its own code.
- Our `HailoFacePipeline` (in `hailo_infer.py`) is configured to
  match.

With matching group_id + scheduling, HailoRT multiplexes the
physical NPU between the two VDevices. Without it, the second one
fails with `HAILO_OUT_OF_PHYSICAL_DEVICES (74)`.

In the common case (OpenAI for chat, Hailo for face + Whisper):

- Face recognition and Whisper rarely overlap anyway. The camera
  worker **skips** recognition during a chat exchange (the
  `chat_busy` check in `main.py`), so the NPU is exclusively
  Whisper's while the user is talking. Whisper completes in
  <1 s and face recognition resumes immediately.

If you also load Hailo-Ollama, the NPU schedules all three workloads
serially. The kiosk doesn't pre-load Ollama when `OPENAI_API_KEY`
is set, so usually you don't pay this cost.

## Troubleshooting

- **Missing HEFs** — the wrapper prints the expected paths and the
  install command. Either move/symlink hailo-apps's downloads, or
  set `HAILO_WHISPER_ENCODER_HEF` / `HAILO_WHISPER_DECODER_HEF` /
  `HAILO_WHISPER_NPY_DIR` to point at them.
- **`ImportError: hailo_apps...whisper_pipeline`** — install the
  speech-rec extra. Clone `hailo-ai/hailo-apps` and run
  `pip install -e '.[speech-rec]'` inside the venv -- it isn't on
  PyPI.
- **HailoRT busy errors at decode** — another model holds the
  NPU. Check `pgrep -fl hailo-ollama` and stop it if you don't need
  offline chat.
- **`add_embed` mismatch** — symptom is gibberish output. Try
  flipping `HAILO_WHISPER_ADD_EMBED` for your hardware.

## Links

- [hailo-ai/hailo-apps](https://github.com/hailo-ai/hailo-apps) —
  primary repo, `python/standalone_apps/speech_recognition/`.
- [Pi 5 + Whisper-Small HEF community thread](https://community.hailo.ai/t/pi-5-whisper-small-hef/18946) —
  troubleshooting + benchmarks from real Pi 5 users.
- [hailocs/hailo-whisper](https://github.com/hailocs/hailo-whisper) —
  original conversion tooling (kept for compatibility).
