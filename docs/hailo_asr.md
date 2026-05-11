# Speech recognition on the Hailo-10H — deferred research

This kiosk currently runs free-form speech recognition (the chat tab's
voice input) on the **CPU** via faster-whisper. The Hailo-10H accelerator
is reserved for vision (SCRFD + ArcFace) and, separately, the
Hailo-Ollama LLM daemon. A natural follow-up is asking whether ASR
could also run on the Hailo chip. The short answer is **not yet** —
the longer answer is below.

## Why not today

1. **No official model.** The Hailo Model Zoo
   (https://github.com/hailo-ai/hailo_model_zoo) does not ship a
   compiled Whisper / Wav2Vec / streaming ASR HEF as of HailoRT 5.3.

2. **Compilation requires the Dataflow Compiler (DFC).** Producing a
   working HEF from Whisper's PyTorch checkpoint needs Hailo's
   licensed compiler toolchain. Community ports exist (search
   `hailo-whisper` on GitHub) but they target older HailoRT versions
   and have rough edges around encoder / decoder split, KV cache,
   and streaming.

3. **Chip contention.** The Hailo-10H is already serving
   `qwen3:1.7b` via Hailo-Ollama for chat replies. The accelerator
   can multiplex models, but inference is sequential per VDevice — so
   transcribing while the LLM is mid-response would either preempt
   the chat (visible latency) or serialise (no benefit over CPU).

4. **CPU is fine.** faster-whisper tiny.en transcribes a 10-second
   utterance in ~3–4 s on the Pi 5's Cortex-A76 cluster, using ~250 MB
   of resident memory. That's already inside the kiosk's response
   budget (the user is talking; we have time to think).

## What would need to happen

For someone picking this back up later:

1. **Pick a model.** Whisper tiny / base are the obvious candidates.
   Distil-Whisper is faster but currently English-only.

2. **Compile to HEF.** Either wait for an official Hailo Model Zoo
   release or use the DFC with one of the community recipes. Expect
   to spend time on tokenisation / chunking / KV-cache plumbing.

3. **Schedule with the LLM.** Decide whether ASR and LLM share one
   VDevice (sequential, simpler) or run on separate VDevices
   (parallel — uses more chip resources but better latency). The
   Hailo-10H has 26 TOPS; both fit, but you must size carefully.

4. **Wire into `chat_voice.py`.** Add a `HailoWhisperASR` class
   alongside `FasterWhisperASR` and select via
   `config.CHAT_ASR_BACKEND = "hailo-whisper"`. The interface
   (`transcribe(pcm: bytes, samplerate: int) -> str`) is already
   designed for this.

5. **Benchmark.** Compare against the CPU baseline. The whole point of
   moving to Hailo is sub-second transcription; if you're stuck at 2 s
   the change isn't worth the complexity.

## Benchmark template

When you do the work, drop a script in `utils/bench_asr.py` that:

- loads a fixed 10-second WAV from `tests/fixtures/`
- runs each backend N=20 times
- prints mean / p50 / p95 latency and resident memory

That gives an objective number to compare against the current
~3 s faster-whisper baseline.

## Links

- Hailo Model Zoo: https://github.com/hailo-ai/hailo_model_zoo
- HailoRT release notes: https://hailo.ai/developer-zone/documentation/
- faster-whisper benchmarks on Pi 5:
  https://github.com/SYSTRAN/faster-whisper#benchmark
- Community Whisper-on-Hailo experiments: search GitHub for
  `hailo whisper`.
