#!/usr/bin/env bash
# Render slides in parallel, one scene per file.
# Usage: scripts/render.sh <quality: l|m|h> [slides/sNN_name.py ...]     (no files = every slide)
# Parallel jobs can race on Manim's shared LaTeX cache, so each job gets a private TeX dir and
# failures are retried serially.
set -uo pipefail
cd "$(dirname "$0")/.."
q="$1"; shift
files=("$@")
[ ${#files[@]} -eq 0 ] && mapfile -t files < <(ls slides/s[0-9][0-9]_*.py)
mkdir -p media/logs
render() {  # $1 = slide file
  local f="$1" s; s=$(grep -oE '^class (S[0-9]{2}[A-Za-z0-9]+)' "$f" | awk '{print $2}')
  mkdir -p "media/tex/$s"
  printf '[CLI]\ntex_dir = media/tex/%s\n' "$s" > "media/logs/$s.cfg"
  manim -q"$q" --progress_bar none --disable_caching --config_file "media/logs/$s.cfg" "$f" "$s" > "media/logs/$s.log" 2>&1
}
export -f render; export q
printf '%s\n' "${files[@]}" | xargs -P "${JOBS:-4}" -I{} bash -c 'render {} || echo {}' > media/logs/.failed
status=0
while read -r f; do
  [ -z "$f" ] && continue
  if render "$f"; then echo "OK   $f (retry)"; else echo "FAIL $f (see media/logs/)"; status=1; fi
done < media/logs/.failed
for f in "${files[@]}"; do grep -qx "$f" media/logs/.failed || echo "OK   $f"; done
exit $status
