#!/usr/bin/env bash
# Download Piper voice models into ./models/piper/.
#
# Default set: a small curated list of clear-sounding English voices.
# Use --all to grab every English voice listed in the script.
# Use --langs "en es fr" to grab a different language set (must match
# Piper's voice IDs; see https://huggingface.co/rhasspy/piper-voices).
#
# After download, switch voices by editing PIPER_MODEL_PATH in config.py
# to point at any of the .onnx files under models/piper/.

set -euo pipefail

PI5_DIR="${PI5_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DEST="$PI5_DIR/models/piper"
HF_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"

mkdir -p "$DEST"

# Curated voices: clear, well-balanced for greeting use.
# en_US-hfc_female-medium is the project default (config.PIPER_MODEL_PATH).
CURATED=(
    # US English - female
    "en/en_US/hfc_female/medium/en_US-hfc_female-medium"
    "en/en_US/amy/medium/en_US-amy-medium"
    "en/en_US/kathleen/low/en_US-kathleen-low"
    "en/en_US/lessac/medium/en_US-lessac-medium"
    # US English - male
    "en/en_US/ryan/medium/en_US-ryan-medium"
    "en/en_US/joe/medium/en_US-joe-medium"
    # UK English
    "en/en_GB/alan/medium/en_GB-alan-medium"
    "en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium"
)

# Larger list if --all is passed.
ALL_EN=(
    "${CURATED[@]}"
    "en/en_US/amy/low/en_US-amy-low"
    "en/en_US/danny/low/en_US-danny-low"
    "en/en_US/hfc_male/medium/en_US-hfc_male-medium"
    "en/en_US/kristin/medium/en_US-kristin-medium"
    "en/en_US/kusal/medium/en_US-kusal-medium"
    "en/en_US/l2arctic/medium/en_US-l2arctic-medium"
    "en/en_US/libritts/high/en_US-libritts-high"
    "en/en_US/libritts_r/medium/en_US-libritts_r-medium"
    "en/en_US/lessac/low/en_US-lessac-low"
    "en/en_US/lessac/high/en_US-lessac-high"
    "en/en_US/ryan/low/en_US-ryan-low"
    "en/en_US/ryan/high/en_US-ryan-high"
    "en/en_GB/alba/medium/en_GB-alba-medium"
    "en/en_GB/aru/medium/en_GB-aru-medium"
    "en/en_GB/cori/medium/en_GB-cori-medium"
    "en/en_GB/cori/high/en_GB-cori-high"
    "en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium"
    "en/en_GB/semaine/medium/en_GB-semaine-medium"
    "en/en_GB/southern_english_female/low/en_GB-southern_english_female-low"
    "en/en_GB/vctk/medium/en_GB-vctk-medium"
)

mode="curated"
case "${1:-}" in
    --all|-a)        mode="all" ;;
    --curated|"")    mode="curated" ;;
    --list|-l)
        echo "Curated set:"
        printf '  %s\n' "${CURATED[@]}"
        echo
        echo "All English (--all):"
        printf '  %s\n' "${ALL_EN[@]}"
        exit 0
        ;;
    --help|-h)
        sed -n '2,12p' "$0"
        exit 0
        ;;
    *)
        echo "Unknown option: $1" >&2
        echo "Usage: $0 [--curated|--all|--list]" >&2
        exit 1
        ;;
esac

case "$mode" in
    curated) voices=("${CURATED[@]}") ;;
    all)     voices=("${ALL_EN[@]}") ;;
esac

echo "Downloading ${#voices[@]} voice(s) to $DEST"
echo "----------------------------------------------------------------"

cd "$DEST"
for path in "${voices[@]}"; do
    name="$(basename "$path")"
    onnx="${name}.onnx"
    json="${name}.onnx.json"

    if [[ -f "$onnx" && -f "$json" ]]; then
        echo "[skip] $name (already present)"
        continue
    fi
    echo "[get ] $name"
    wget -q --show-progress "$HF_BASE/$path.onnx"      -O "$onnx" || { rm -f "$onnx"; echo "  failed: .onnx"; continue; }
    wget -q --show-progress "$HF_BASE/$path.onnx.json" -O "$json" || { rm -f "$json"; echo "  failed: .onnx.json"; continue; }
done

echo
echo "Done. Voices in $DEST:"
ls -lh "$DEST"/*.onnx 2>/dev/null | awk '{print "  "$NF, "("$5")"}'
echo
echo "To switch voice, edit config.py:"
echo "  PIPER_MODEL_PATH = MODELS_DIR / \"piper\" / \"<voice>.onnx\""
