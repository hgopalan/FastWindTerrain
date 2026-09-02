# Thomas Fire

California, 2017 — 71,481 ha over 39 days.

A 5 × 5 km domain centred on 34.3667, -119.0667, at 50 m horizontal
resolution (100 × 100 × 60 cells).

**Why this case is in the catalogue:** The Ventura / Santa Ynez front — Santa Ana and sundowner winds over steep terrain.

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
