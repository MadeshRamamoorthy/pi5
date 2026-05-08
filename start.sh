#!/usr/bin/env bash
# Start the face-recognition app with the right mic + speaker pinned.
#
# Override any of these by exporting them before running, or by editing
# the defaults below. The script also forwards any extra args you pass
# straight through to main.py (e.g. ./start.sh --no-display).
#
#   SD_DEVICE             sounddevice index for the mic.
#                         List with: python debug_mic.py --list
#   AUDIO_OUTPUT_DEVICE   ALSA name for the speaker.
#                         List with: aplay -L
#   PI5_DIR               project directory.
#   EXTRA_ARGS            default args passed to main.py.
#
# Examples:
#   ./start.sh
#   ./start.sh --no-display
#   SD_DEVICE=3 ./start.sh
#   EXTRA_ARGS="--no-wake-word --auto-register" ./start.sh

set -euo pipefail

# ---- Defaults (edit these once for your machine) -------------------------
: "${PI5_DIR:=/home/echo/Documents/code/pi5}"
: "${SD_DEVICE:=1}"
: "${AUDIO_OUTPUT_DEVICE:=plughw:CARD=PowerConf,DEV=0}"
: "${EXTRA_ARGS:=--auto-register}"

# Silence the harmless cv2/Qt font warning (main.py also sets this; doing
# it here too means the warning is gone even before Python starts).
export QT_LOGGING_RULES="qt.qpa.fonts.warning=false"

# ---- Sanity checks -------------------------------------------------------
cd "$PI5_DIR"

if [[ ! -f .venv/bin/activate ]]; then
    echo "Error: .venv not found at $PI5_DIR/.venv" >&2
    echo "Set up the venv first (see README §2.5)." >&2
    exit 1
fi

if [[ ! -f models/scrfd_10g.hef || ! -f models/arcface_mobilefacenet.hef ]]; then
    echo "Error: HEFs missing in $PI5_DIR/models/" >&2
    echo "Download them (see README §2.7)." >&2
    exit 1
fi

# Optional: warn if Vosk model dir is missing -- we still proceed because
# main.py falls back to ACTIVE state without the wake word.
if [[ ! -d models/vosk-model-small-en-us-0.15 ]]; then
    echo "Note: Vosk model not found; wake-word will be disabled." >&2
    echo "      Download (see README §2.8) to enable 'hello echo'." >&2
fi

# ---- Display / X handling -----------------------------------------------
# Common pitfall: running this script as root drops the user's X session
# variables, so cv2's Qt preview can't connect to the display. Fix it up
# here automatically. If we still can't reach an X display, fall back to
# --no-display so the app at least runs headless.
if [[ $EUID -eq 0 ]]; then
    echo "Warning: running as root. Prefer running as the 'echo' user." >&2
    if [[ -z "${DISPLAY:-}" ]]; then
        export DISPLAY=":0"
    fi
    if [[ -z "${XAUTHORITY:-}" && -f /home/echo/.Xauthority ]]; then
        export XAUTHORITY=/home/echo/.Xauthority
    fi
fi

x_ok=true
if ! command -v xset >/dev/null 2>&1; then
    # No xset; we can't probe. Trust DISPLAY being set.
    [[ -z "${DISPLAY:-}" ]] && x_ok=false
elif ! xset q >/dev/null 2>&1; then
    x_ok=false
fi

if ! $x_ok; then
    echo "Note: no usable X display; adding --no-display." >&2
    case " $EXTRA_ARGS $* " in
        *" --no-display "*) ;;
        *) EXTRA_ARGS="$EXTRA_ARGS --no-display" ;;
    esac
fi

# ---- Activate venv & launch ---------------------------------------------
# shellcheck source=/dev/null
source .venv/bin/activate

export SD_DEVICE
export AUDIO_OUTPUT_DEVICE

echo "---------------------------------------------------------------"
echo " pi5 face-recognition"
echo "   project    : $PI5_DIR"
echo "   mic        : sounddevice index $SD_DEVICE"
echo "   speaker    : $AUDIO_OUTPUT_DEVICE"
echo "   args       : $EXTRA_ARGS $*"
echo "---------------------------------------------------------------"

exec python main.py $EXTRA_ARGS "$@"
