#!/usr/bin/env bash
# Render slide scenes in parallel.  Usage: scripts/render.sh <quality: l|m|h> <file.py> [Scene ...]
# With no scene names, every class starting with "S" + two digits in the file is rendered.
# Parallel jobs can race on Manim's shared LaTeX cache, so failures are retried serially.
set -uo pipefail
cd "$(dirname "$0")/.."
q="$1"; file="$2"; shift 2
scenes=("$@")
if [ ${#scenes[@]} -eq 0 ]; then
  mapfile -t scenes < <(grep -oE '^class (S[0-9]{2}[A-Za-z0-9]+)' "$file" | awk '{print $2}')
fi
mkdir -p media/logs
render() {  # private LaTeX dir per job avoids most cache races
  mkdir -p "media/tex/$1"
  printf '[CLI]\ntex_dir = media/tex/%s\n' "$1" > "media/logs/$1.cfg"
  manim -q"$q" --progress_bar none --disable_caching --config_file "media/logs/$1.cfg" "$file" "$1" > "media/logs/$1.log" 2>&1
}
export -f render; export q file
printf '%s\n' "${scenes[@]}" | xargs -P "${JOBS:-4}" -I{} bash -c 'render {} || echo {}' > media/logs/.failed
status=0
while read -r s; do
  [ -z "$s" ] && continue
  if render "$s"; then echo "OK   $s (retry)"; else echo "FAIL $s (see media/logs/$s.log)"; status=1; fi
done < media/logs/.failed
for s in "${scenes[@]}"; do grep -qx "$s" media/logs/.failed || echo "OK   $s"; done
exit $status
