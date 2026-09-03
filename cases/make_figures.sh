#!/usr/bin/env bash
#
# make_figures.sh -- regenerate every phase 22a figure.
#
# Writes into data/figures, which .gitignore already covers along with the
# rest of data/. The figures are a few megabytes and take a couple of
# minutes to rebuild, so they are regenerated rather than committed --
# and regenerating them is also how you check they still describe the
# dataset on disk rather than an older one.
#
# Needs the datasets (cases/build_dataset.py) and matplotlib:
#
#     pip install -e '.[figures]'
#
# Usage:
#
#     bash cases/make_figures.sh                  # into data/figures
#     bash cases/make_figures.sh /somewhere/else
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/data/figures}"
export PYTHONPATH="$ROOT/build/python:$ROOT/cases${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
mkdir -p "$OUT"

if [ ! -d data/corpus ]; then
    echo "no data/corpus -- run cases/build_dataset.py first" >&2
    exit 1
fi

# The windows are chosen to span the corpus rather than to look good:
# the steepest window in the test fold, the gentlest, and one from each
# demonstration site. 45 degrees because the dataset solves 0/45/90/135
# and the other four directions are exact negations.
STEEP=ditch_fire:10          # 1970 m relief, the corpus maximum
GENTLE=marshall_fire:10      #   86 m relief, the minimum

echo "== per-level histograms =="
python3 cases/error_maps.py --histogram                --out "$OUT"
python3 cases/error_maps.py --histogram --what baseline --out "$OUT"

echo "== error against terrain slope =="
python3 cases/slope_error.py                --out "$OUT"
python3 cases/slope_error.py --what baseline --out "$OUT"

echo "== per-level maps =="
for w in "$STEEP" "$GENTLE"; do
    python3 cases/error_maps.py --window "$w" --direction 45 --limit 1 \
        --out "$OUT"
    python3 cases/error_maps.py --window "$w" --direction 45 --limit 1 \
        --what baseline --out "$OUT"
done

if [ -d data/demo ]; then
    echo "== demonstration sites =="
    for w in cameron_peak:12 chetco_bar:12; do
        python3 cases/error_maps.py --data data/demo --fold demo \
            --window "$w" --direction 45 --limit 1 --out "$OUT"
    done
else
    echo "no data/demo -- skipping the demonstration sites" >&2
fi

echo
echo "figures in $OUT:"
ls -1 "$OUT"
