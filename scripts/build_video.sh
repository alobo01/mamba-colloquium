#!/usr/bin/env bash
# Render every slide at one quality and concatenate them, in talk order, into one MP4.
# Usage: scripts/build_video.sh [l|m|h]   (default m = 720p30)
set -uo pipefail
cd "$(dirname "$0")/.."
q="${1:-m}"
case "$q" in l) dir=480p15 ;; m) dir=720p30 ;; h) dir=1080p60 ;; *) echo "quality must be l, m or h"; exit 2 ;; esac
files=(slides/act1_attention.py slides/act2_notebooks.py slides/act3_ssd.py slides/act4_clinic_belt.py slides/act5_theory.py)
python -m pytest -q math_contracts || { echo "math contract failed: not rendering"; exit 1; }
status=0
for f in "${files[@]}"; do
  [ -n "${SKIP_RENDER:-}" ] || scripts/render.sh "$q" "$f" || status=1
done
mkdir -p media/final
list=media/final/concat_$dir.txt
: > "$list"
for f in "${files[@]}"; do
  stem=$(basename "$f" .py)
  for s in $(grep -oE '^class (S[0-9]{2}[A-Za-z0-9]+)' "$f" | awk '{print $2}'); do
    mp4="media/videos/$stem/$dir/$s.mp4"
    if [ -f "$mp4" ]; then echo "file '$(pwd)/$mp4'" >> "$list"; else echo "missing $mp4"; status=1; fi
  done
done
out=media/final/three_brothers_$dir.mp4
ffmpeg -loglevel error -y -f concat -safe 0 -i "$list" -c copy -movflags +faststart "$out" && echo "wrote $out"
exit $status
