# Offline tests

Fast checks (under a minute, no internet) for the shared engine and every kit's core methods.

```
python tests/run_all.py              # all tests
python -m unittest test_health -v    # one file (run from inside tests/)
```

Use the Python that ships with QGIS (it has GDAL, numpy, matplotlib and Qt), or any Python with those installed.
On Windows: the OSGeo4W Shell; on macOS: `/Applications/QGIS.app/Contents/MacOS/bin/python3`.

| File | What it checks |
|---|---|
| `test_health.py` | Byar CI vs exact Poisson, SMR/expected cases, empirical Bayes shrinkage, Moran's I vs a dense-matrix formula, LISA sums to global I (PySAL convention), Benjamini–Hochberg FDR, Kulldorff scan finds a planted cluster, BYM2 smoke run |
| `test_methods.py` | Flood HAND stage calibration and design depth, NOAA heat-index table, Mann–Kendall/Sen slope, crop phenology (0/1/2 seasons on synthetic curves), landslide runout stops at the 11° energy line |
| `test_infra.py` | Kit registry and `plugin_info` filtering, command line, download retries on HTTP 429 and the on-disk cache (local test server), report written as HTML, Word and PDF with non-ASCII text |

The full end-to-end runs on real data (Dadu, Gurugram, Hiroshima, Kentucky, North Carolina) live in
`validation/` outside the plugin and need internet; they take 3–30 minutes each.

This folder is for development only and should not be shipped inside the plugin zip.
