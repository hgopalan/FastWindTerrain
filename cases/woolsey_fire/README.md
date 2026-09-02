# Woolsey Fire

California, 2018 — 24,305 ha over 9 days.

A 5 × 5 km domain centred on 34.2667, -118.7833, at 50 m horizontal
resolution (100 × 100 × 60 cells).

**Why this case is in the catalogue:** The Santa Monica Mountains — a Santa Ana corridor that has burned repeatedly.

## Use it

```
python3 prepare.py          # download SRTM, write terrain.csv and survey.json
python3 run.py              # solve, 8 m/s from the southwest
```

`terrain.csv` is a megabyte or so and is **not committed**. `survey.json` is,
and the vertical grid is derived from it — so the grid this case uses is
reproducible without a download, and `prepare.py --no-download` rebuilds it.

The domain floor sits at the tile's minimum elevation and the top a
kilometre above its highest ground, so the vertical extent follows this
case's own relief. Elevations are absolute metres above sea level;
horizontal coordinates are local metres on [0, 5000].
