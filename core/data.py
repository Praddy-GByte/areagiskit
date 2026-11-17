# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Open data access: STAC search, URL signing and clip-on-read.

Everything here works without an account or API key:
  * Microsoft Planetary Computer (anonymous SAS signing)
  * public cloud buckets (e.g. ASF GLO-30 HAND on AWS)
Only the pixels inside the area of interest are streamed; nothing is
downloaded in full.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="
UA = "AreaGIS-Kit/0.14 (open-source GIS plugin)"


class KitError(Exception):
    """An error whose message is written for the end user."""


CACHE_TTL = 6 * 3600           # default life of a cached JSON GET answer (seconds); a rerun the same day reuses it
_RETRY_CODES = {408, 425, 429, 500, 502, 503, 504}


def _cache_dir():
    import os
    import tempfile
    d = os.path.join(tempfile.gettempdir(), "areagiskit_cache")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return None
    return d


def _cache_path(url):
    import hashlib
    import os
    d = _cache_dir()
    return os.path.join(d, hashlib.sha256(url.encode("utf-8")).hexdigest()[:40] + ".json") if d else None


def _cache_get(url, ttl):
    import os
    p = _cache_path(url)
    if not p or not os.path.exists(p):
        return None
    try:
        if time.time() - os.path.getmtime(p) > ttl:
            return None
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _cache_put(url, obj):
    import os
    p = _cache_path(url)
    if not p:
        return
    tmp = p + ".part"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh)
        os.replace(tmp, p)          # atomic on Windows and POSIX
    except (OSError, TypeError, ValueError):
        try:
            os.remove(tmp)
        except OSError:
            pass


def _retry_after(exc, attempt):
    """Seconds to wait: the server's Retry-After header if given (capped), else exponential back-off."""
    try:
        ra = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
        if ra:
            return min(float(ra), 60.0)
    except (TypeError, ValueError):
        pass
    return min(2.0 * (2 ** attempt), 30.0)


def get_json(url, body=None, timeout=60, tries=4, cache=None):
    """GET (or POST when `body` is given) a JSON document.

    Retries on time-outs, dropped connections and HTTP 408/425/429/5xx with back-off (honouring Retry-After).
    GET answers are cached on disk for `cache` seconds (default CACHE_TTL) so a rerun after a failure does not
    hit the same APIs again; pass cache=0 for live data such as forecasts or signing tokens. POSTs are never cached."""
    ttl = CACHE_TTL if cache is None else cache
    use_cache = body is None and ttl and ttl > 0
    if use_cache:
        hit = _cache_get(url, ttl)
        if hit is not None:
            return hit
    data = json.dumps(body).encode() if body is not None else None
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={
                "Content-Type": "application/json", "User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                obj = json.loads(r.read().decode("utf-8"))
            if use_cache:
                _cache_put(url, obj)
            return obj
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in _RETRY_CODES:
                break
            wait = _retry_after(exc, attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as exc:
            last = exc
            wait = min(2.0 * (2 ** attempt), 30.0)
        if attempt < tries - 1:
            time.sleep(wait)
    if isinstance(last, urllib.error.HTTPError) and last.code == 429:
        raise KitError("A free data service is busy (too many requests). Wait a few minutes and run again; "
                       "answers already received are cached, so the rerun is faster.")
    raise KitError(f"Could not reach the data server ({last}). Check the internet connection and try again.")


def iso_range(start, end):
    """'2022-08-01', '2022-09-01' -> STAC datetime interval."""
    return f"{start}T00:00:00Z/{end}T23:59:59Z"


def search(collection, bbox, start, end, query=None, max_items=500):
    body = {"collections": [collection], "bbox": bbox,
            "datetime": iso_range(start, end), "limit": 100}
    if query:
        body["query"] = query
    out, url, b = [], f"{STAC}/search", body
    while True:
        d = get_json(url, b)
        out.extend(d.get("features", []))
        nxt = [l for l in d.get("links", []) if l.get("rel") == "next"]
        if not nxt or len(out) >= max_items:
            break
        n = nxt[0]
        url, b = n["href"], n.get("body", None)
        if n.get("method", "GET") == "GET":
            b = None
    out = out[:max_items]
    if collection == "sentinel-2-l2a":
        out = s2_dedupe(out)
    return out


def _s2_base(it):
    return it["properties"].get("s2:processing_baseline", "00.00") or "00.00"


def s2_dedupe(items):
    """Planetary Computer can list the same Sentinel-2 tile and date twice: the original processing and an ESA
    reprocessing (baseline ≥ 05.00, which adds a +1000 reflectance offset). Keep only the newest processing of each."""
    best = {}
    for it in items:
        p = it["properties"]
        parts = it["id"].split("_")
        tile = p.get("s2:mgrs_tile") or (parts[5] if len(parts) > 5 else it["id"])
        k = (tile, (p.get("datetime") or "")[:19])
        if k not in best or _s2_base(it) > _s2_base(best[k]):
            best[k] = it
    keep = set(id(v) for v in best.values())
    return [it for it in items if id(it) in keep]


def s2_offset(it):
    return 1000.0 if _s2_base(it) >= "04.00" else 0.0


def warp_s2(items, band, grid, resample="average"):
    """Surface reflectance (0–1) of a Sentinel-2 L2A band, with the offset of each item's own processing baseline
    removed (items of one day can mix baselines when one tile was reprocessed and another was not)."""
    groups = {}
    for it in items:
        groups.setdefault(s2_offset(it), []).append(it)
    out = None
    for off, its in groups.items():
        a = (warp_items(its, band, grid, resample=resample, dtype=gdal.GDT_Float32, src_nodata=0, dst_nodata=np.nan) - off) / 1e4
        out = a if out is None else np.where(np.isfinite(out), out, a)
    return out


def item_date(item):
    p = item["properties"]
    return (p.get("datetime") or p.get("start_datetime") or "")[:10]


# Provenance: every scene/tile actually read in this run, so reports can list and link them.
TRACE = []
STAC_ITEM = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/{c}/items/{i}"


def trace_reset():
    TRACE.clear()


def trace(items=None, url=None, label=None):
    for it in items or []:
        rec = (it.get("collection", "?"), it["id"], item_date(it),
               STAC_ITEM.format(c=it.get("collection", ""), i=it["id"]))
        if rec not in TRACE:
            TRACE.append(rec)
    if url:
        rec = (label or "file", url.rsplit("/", 1)[-1], "", url)
        if rec not in TRACE:
            TRACE.append(rec)


_SAS = {}


def sign(href):
    """Return a readable URL. Planetary Computer blobs get a cached SAS token."""
    u = urllib.parse.urlparse(href)
    if not u.netloc.endswith("blob.core.windows.net"):
        return href
    key = u.netloc + "/" + u.path.split("/")[1]
    tok = _SAS.get(key)
    if not tok or tok[1] < time.time() + 120:
        d = get_json(SIGN + urllib.parse.quote(href, safe=""), cache=0)   # tokens expire: never cache
        q = urllib.parse.urlparse(d["href"]).query
        _SAS[key] = (q, time.time() + 40 * 60)
        tok = _SAS[key]
    return href + "?" + tok[0]


def vsi(href):
    return "/vsicurl/" + sign(href)


def configure_gdal():
    for k, v in {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "3",
        "GDAL_HTTP_RETRY_CODES": "429,500,502,503,504",
        "GDAL_HTTP_TIMEOUT": "120",
        "GDAL_HTTP_CONNECTTIMEOUT": "30",
        "CPL_VSIL_CURL_CHUNK_SIZE": "524288",
        "VSI_CACHE": "TRUE",
        "GDAL_CACHEMAX": "512",
    }.items():
        gdal.SetConfigOption(k, v)


def warp(sources, grid, resample="near", dtype=gdal.GDT_Float32,
         src_nodata=None, dst_nodata=None):
    """Read one or many rasters onto the common grid. Returns a numpy array."""
    sources = [s for s in sources if s]
    if not sources:
        raise KitError("No data found for this area and date.")
    kw = dict(format="MEM", outputBounds=grid.bounds, xRes=grid.res, yRes=grid.res,
              dstSRS=grid.wkt, resampleAlg=resample, outputType=dtype,
              multithread=True, warpMemoryLimit=512)
    if src_nodata is not None:
        kw["srcNodata"] = src_nodata
    if dst_nodata is not None:
        kw["dstNodata"] = dst_nodata
    ds = gdal.Warp("", sources, **kw)
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    return arr


def warp_items(items, asset, grid, **kw):
    trace(items)
    return warp([vsi(it["assets"][asset]["href"]) for it in items if asset in it["assets"]], grid, **kw)


def exists(url):
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status == 200
    except Exception:
        return False
