#!/usr/bin/env bash
# Build a contact sheet (grid of frames) from a rendered slide for visual QA.
# Usage: scripts/contact_sheet.sh video.mp4 out.png [frames=12]
set -euo pipefail
v="$1"; out="$2"; n="${3:-12}"
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$v")
fps=$(python3 -c "print($n/$dur)")
ffmpeg -loglevel error -y -i "$v" -vf "fps=$fps,scale=480:-1,tile=4x$(( (n+3)/4 ))" -frames:v 1 "$out"
