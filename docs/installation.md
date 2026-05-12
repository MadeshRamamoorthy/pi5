# Installation

For a fresh Pi 5 with no prior setup, work through the sections in
order.

| # | Section | Time | Output |
|---|---|---|---|
| 1 | [OS prep + PCIe Gen 3](#1-os-prep--pcie-gen-3) | 5 min + reboot | apt deps, Gen 3 PCIe enabled |
| 2 | [Hailo driver + runtime](#2-hailo-driver--runtime) | 5 min | HailoRT 5.x, `hailortcli` works |
| 3 | [Clone + venv](#3-clone--venv) | 3 min | running Python venv with deps |
| 4 | [Models](#4-models) | 10 min | HEFs + Vosk + Piper voices in `models/` |
| 5 | [Audio](#5-audio) | 2 min | mic + speaker verified |
| 6 | [Browser](#6-browser) | 2 min | Chromium kiosk installed |

---

## 1. OS prep + PCIe Gen 3

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3.13-venv python3-pip git \
                    libportaudio2 portaudio19-dev \
                    espeak-ng alsa-utils \
                    ffmpeg                          # required by hailo-apps speech-rec
sudo raspi-config nonint do_pcie_gen 3              # Hailo-10H needs Gen 3
sudo reboot
```

## 2. Hailo driver + runtime

apt's `hailo-all` ships HailoRT 4.23 which doesn't know about
Hailo-10H. Grab HailoRT 5.x as a `.deb` from
<https://hailo.ai/developer-zone/software-downloads/>:

- `hailort_5.x.y_arm64.deb`
- `hailort-5.x.y-cp313-cp313-linux_aarch64.whl` (matches Trixie's
  Python 3.13; pick `cp311` for Bookworm)

**Don't** install Hailo's `hailort-pcie-driver` `.deb` — apt's
`hailo-all` already supplies the matching kernel module.

```bash
sudo cp ~/Downloads/hailort_5.*_arm64.deb /tmp/
sudo apt install /tmp/hailort_5.*_arm64.deb

hailortcli --version                                # must show 5.x
hailortcli fw-control identify                      # Architecture: HAILO10H
```

If `apt update` flags a stale `hailo.list` source:

```bash
sudo rm -f /etc/apt/sources.list.d/hailo.list /etc/apt/keyrings/hailo.gpg
```

## 3. Clone + venv

```bash
mkdir -p ~/Documents/code && cd ~/Documents/code
git clone <your-fork-url> pi5
cd pi5

python3 -m venv --system-site-packages .venv      # for apt-installed picamera2/libcamera
source .venv/bin/activate
pip install --upgrade pip
pip install ~/Downloads/hailort-5.*-cp313-cp313-linux_aarch64.whl
pip install -r requirements.txt
```

`--system-site-packages` exposes the apt-installed `picamera2` +
`libcamera` to the venv. Without it you'll see `ImportError` at
startup.

> **Heads-up on PEP 668.** On Trixie, running `pip install` *outside*
> the venv now fails with `error: externally-managed-environment`.
> Always activate the venv first (`source .venv/bin/activate`) before
> any `pip install`. If you ever genuinely need to install into the
> system Python, use `pip install --break-system-packages` — but for
> this project, the venv is what you want.

## 4. Models

All required + optional files live in `models/`.

| File | Purpose | Where to get it |
|---|---|---|
| `scrfd_10g.hef` | Face detector (Hailo) | <https://github.com/hailo-ai/hailo_model_zoo/releases> · filter HAILO10H |
| `arcface_mobilefacenet.hef` | Face embedder (Hailo) | same source |
| `vosk-model-small-en-us-0.15/` | Wake-word recogniser | `wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip` and unzip into `models/` |
| `whisper-base-encoder.hef`, `whisper-base-decoder.hef`, `whisper-base-assets/` | Hailo Whisper STT (recommended) | Clone `hailo-ai/hailo-apps` and `pip install -e '.[speech-rec]'` (it's not on PyPI). Hailo's own CLI auto-downloads the assets on first run. Symlink or copy into `models/`. See [Hailo Whisper](hailo-whisper.md) |
| `en_US-hfc_female-medium.onnx` + `.json` | Piper TTS voice | `python -m piper.download_voices en_US-hfc_female-medium --data-dir models/` |

### Verify Hailo HEF compatibility

```bash
hailortcli parse-hef models/scrfd_10g.hef             | head -1
hailortcli parse-hef models/arcface_mobilefacenet.hef | head -1
# Both must say: HEF Compatible for: HAILO10H
```

### Smoke-test the HEF

```bash
hailortcli run2 -t 5 set-net models/scrfd_10g.hef
# expect fps: 240+
```

Note: `run2` is the H10 path. The older `run` subcommand won't work.

## 5. Audio

```bash
arecord -l                                          # find a capture device
arecord -d 3 -f cd /tmp/test.wav && aplay /tmp/test.wav    # mic loopback
```

If you have a USB speakerphone you want to pin (instead of HDMI):

**Option A — make it the system default.**

PipeWire (Trixie default):

```bash
sudo apt install -y wireplumber
wpctl status                                      # find the sink ID
wpctl set-default <SINK_ID>
```

PulseAudio compat layer:

```bash
sudo apt install -y pulseaudio-utils
pactl list short sinks
pactl set-default-sink <sink-name>
```

`raspi-config` → System → Audio also works for HDMI / headphone jack.

**Option B — pin only this app's TTS.**

In `.env`:

```
AUDIO_OUTPUT_DEVICE=plughw:CARD=PowerConf,DEV=0
```

Find the ALSA device name via `aplay -l`:

```
card 2: PowerConf [PowerConf S3], device 0: USB Audio [USB Audio]
                ^^^^^^^^^^^^^^^^^
                use this in plughw:CARD=...
```

## 6. Browser

```bash
sudo apt install -y chromium fonts-noto-color-emoji
```

The kiosk SPA uses emoji in headers (👋, 🌡️, 💧). Without
`fonts-noto-color-emoji` Chromium renders them as monochrome squares.

`start.sh` tries `chromium-browser`, `chromium`, `google-chrome`,
`firefox` in order — first one found wins.

---

Once everything above is done, head to [Quickstart](quickstart.md).
