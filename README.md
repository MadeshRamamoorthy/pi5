# Pi 5 + Hailo-10H Face Recognition

Live face recognition on a Raspberry Pi 5 (8 GB) using the **Pi AI Camera** as
image source and a **Hailo-10H M.2** accelerator running detection (SCRFD) +
embedding (ArcFace MobileFaceNet). Recognised employees are greeted by name
through TTS; unknown faces trigger an interactive registration that stores the
person in a local SQLite database.

## How it works

```
 Pi AI Camera ──▶ picamera2 ──▶ BGR frame
                                   │
                                   ▼
                        Hailo-10H: SCRFD face detect
                                   │ (bbox + 5 landmarks)
                                   ▼
                        Align to 112×112 (ArcFace ref)
                                   │
                                   ▼
                  Hailo-10H: ArcFace embedding (512-D)
                                   │
                                   ▼
                Cosine match vs SQLite (faces.db)
                       │                     │
                  match ≥ 0.45           below threshold
                       │                     │
                       ▼                     ▼
                pyttsx3 greet         Prompt to register
                                      (emp_id, name) → DB
```

The DB has two tables: `employees(emp_id, name)` and
`face_embeddings(emp_id, embedding BLOB)`. Multiple embeddings per person are
allowed (recommended) — each enrolment shot is one row.

## Project layout

| File | Purpose |
|------|---------|
| `config.py`        | Paths, thresholds, camera settings |
| `database.py`      | SQLite schema + CRUD for employees / embeddings |
| `hailo_infer.py`   | HailoRT pipeline: SCRFD decode + NMS, ArcFace embed, alignment |
| `main.py`          | Live loop: detect → embed → match → greet / register |
| `enroll.py`        | Batch-enroll an employee from N camera frames |
| `requirements.txt` | Python deps (HailoRT is installed system-wide, not via pip) |

---

## 1. Hardware checklist

- Raspberry Pi 5 (8 GB) on Bookworm 64-bit
- Active cooler (the Hailo M.2 + Pi 5 will run hot under load)
- Pi AI Camera (IMX500) on the CSI ribbon
- Hailo-10H M.2 module seated in the Pi AI HAT+ / M.2 HAT, PCIe enabled
- Adequate PSU (the official 27 W USB-C is recommended)

---

## 2. Step-by-step install / activation

### 2.1 Update the OS and enable PCIe Gen 3

```bash
sudo apt update && sudo apt full-upgrade -y
sudo rpi-eeprom-update -a       # ensure latest bootloader
```

Edit `/boot/firmware/config.txt` and add (under `[all]`):

```ini
dtparam=pciex1
dtparam=pciex1_gen=3
camera_auto_detect=1
```

Reboot:

```bash
sudo reboot
```

After reboot, confirm the Hailo device is visible on PCIe:

```bash
lspci | grep -i hailo
```

You should see a Hailo Technologies entry.

### 2.2 Install Hailo runtime (HailoRT + PCIe driver + Python bindings)

Raspberry Pi OS provides packaged builds:

```bash
sudo apt install -y hailo-all
sudo reboot
```

`hailo-all` brings in: the PCIe kernel driver (`hailo_pci`), `hailort`
userspace libraries, the `hailortcli` CLI, and the `hailo_platform`
Python module that this project imports.

Verify everything is up:

```bash
hailortcli fw-control identify          # should print a Hailo-10H device
lsmod | grep hailo                      # hailo_pci loaded
python3 -c "import hailo_platform; print(hailo_platform.__version__)"
```

### 2.3 Install Pi AI Camera support

```bash
sudo apt install -y python3-picamera2 python3-libcamera imx500-all
```

Quick sanity check (preview):

```bash
rpicam-hello -t 5000
```

### 2.4 Clone this repo and install Python deps

```bash
mkdir -p /home/echo/Documents/code
cd /home/echo/Documents/code
git clone <your-fork-url> pi5
cd /home/echo/Documents/code/pi5
python3 -m venv --system-site-packages .venv   # so picamera2 + hailo_platform are visible
source .venv/bin/activate
pip install -r requirements.txt
```

> The `--system-site-packages` flag is important: `picamera2` and
> `hailo_platform` are installed at the system level by `apt`, and we want the
> venv to inherit them rather than reinstall.

### 2.5 Install system audio for TTS

`pyttsx3` uses `espeak-ng` under the hood on Linux:

```bash
sudo apt install -y espeak-ng alsa-utils
aplay -l            # confirm an output device exists (HDMI / 3.5mm / USB)
```

### 2.6 Download the Hailo HEF models

This project expects two compiled `.hef` files in `models/`:

- `models/scrfd_10g.hef`           (face detector)
- `models/arcface_mobilefacenet.hef` (face embedder)

Get them from the Hailo Model Zoo. With the `hailo_model_zoo` package
installed (`pip install hailo-model-zoo`) you can grab the pre-compiled HEFs
for the Hailo-10H target:

```bash
mkdir -p /home/echo/Documents/code/pi5/models
cd /home/echo/Documents/code/pi5/models

# Browse https://github.com/hailo-ai/hailo_model_zoo/releases and grab the
# hailo10h HEFs (or download from the Hailo Developer Zone), then verify:
ls -lh scrfd_10g.hef arcface_mobilefacenet.hef
hailortcli run scrfd_10g.hef --measure-fps   # smoke test
```

> The `hailo-model-zoo` package is **not on PyPI** — `pip install
> hailo-model-zoo` will fail. Either install it from
> `https://github.com/hailo-ai/hailo_model_zoo` (`git clone && pip install
> -e .`) or, simpler on a Pi 5, just download the `.hef` files directly
> from the Model Zoo releases page or the Hailo Developer Zone.

If the filenames differ after download, either rename them to match
`config.py` or edit `config.DETECTOR_HEF` / `config.EMBEDDER_HEF`.

> The decoder in `hailo_infer.py` is written for **SCRFD with 9 raw output
> heads** (3 strides × {score, bbox, kps}) and standard ArcFace output. If you
> swap models, adjust the decode helpers accordingly.

---

## 3. Using it

### 3.1 Pre-enrol known employees (recommended)

```bash
python enroll.py --emp-id E001 --name "Alice Kumar" --frames 5
python enroll.py --emp-id E002 --name "Bob Singh"   --frames 5
```

Move your head slightly between frames to capture pose variation.

### 3.2 Run the live recogniser

```bash
python main.py --auto-register
```

- Recognised faces are greeted by name (rate-limited to once per 10 s per
  person — tweak via `GREET_COOLDOWN_SEC` in `config.py`).
- Unknown faces (largest in frame) trigger a console prompt for `emp_id` +
  `name`; the embedding is written to `faces.db` and used for subsequent
  recognition.
- Press **`r`** in the preview window to force-register the largest face.
- Press **`q`** to quit.

Run headless (no preview window):

```bash
python main.py --auto-register --no-display
```

### 3.3 Tuning

All thresholds live in `config.py`:

| Setting | Purpose |
|---------|---------|
| `DETECTOR_SCORE_THRESHOLD` | Lower → more detections, more false positives |
| `COSINE_MATCH_THRESHOLD`   | Lower → looser identity match (more false accepts) |
| `GREET_COOLDOWN_SEC`       | Don't repeat greeting within this window |
| `CAMERA_RESOLUTION`        | Source resolution (downscaled to 640×640 for SCRFD) |

For most ArcFace MobileFaceNet variants, **0.40–0.50** cosine is a sensible
operating range. Calibrate with your own enrolment set if accuracy matters.

---

## 4. Run on boot (optional)

Create a systemd unit at `/etc/systemd/system/face-recog.service`:

```ini
[Unit]
Description=Pi5 Face Recognition
After=multi-user.target

[Service]
Type=simple
User=echo
WorkingDirectory=/home/echo/Documents/code/pi5
ExecStart=/home/echo/Documents/code/pi5/.venv/bin/python /home/echo/Documents/code/pi5/main.py --auto-register --no-display
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now face-recog.service
journalctl -u face-recog.service -f
```

---

## 5. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `hailortcli` says no device | PCIe not enabled in `config.txt`, or M.2 not seated. Check `lspci`. |
| `ImportError: hailo_platform` | Running outside the venv, or venv built without `--system-site-packages`. |
| `Picamera2` cannot find a camera | `camera_auto_detect=1` missing, or the IMX500 ribbon is reversed. |
| Greeting doesn't speak | No audio sink. Run `aplay /usr/share/sounds/alsa/Front_Center.wav`. Set default sink with `raspi-config`. |
| Always says "unknown" | Threshold too strict, or only one enrolment shot — re-enrol with more frames. |
| Recognises the wrong person | Threshold too loose; raise `COSINE_MATCH_THRESHOLD` or enrol more samples per person. |

---

## 6. Privacy note

Face embeddings are biometric data. Store `faces.db` on an encrypted volume
if this leaves a controlled environment, and only enrol people who have
consented. This repo gitignores `faces.db` by default.
