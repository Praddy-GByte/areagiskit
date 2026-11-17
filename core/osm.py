# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""OpenStreetMap facilities via the public Overpass API (no account)."""

import json
import urllib.parse
import urllib.request

from .data import UA, KitError
from .net import open_url

ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]

TAGS = {
    "health": ['["amenity"~"^(hospital|clinic|doctors)$"]', '["healthcare"~"^(hospital|clinic|centre|doctor)$"]'],
    "school": ['["amenity"~"^(school|college)$"]'],
    "all": ['["amenity"~"^(hospital|clinic|doctors|school|college)$"]'],
}


def facilities(bbox, kind="health", buffer_deg=0.05, timeout=25):
    """Return list of (lon, lat, name, type) for facilities in and around the bbox."""
    x0, y0, x1, y1 = bbox
    b = f"{y0 - buffer_deg},{x0 - buffer_deg},{y1 + buffer_deg},{x1 + buffer_deg}"
    parts = "".join(f"nwr{t}({b});" for t in TAGS.get(kind, TAGS["health"]))
    q = f"[out:json][timeout:{timeout}];({parts});out center tags;"
    last = None
    for ep in ENDPOINTS:
        try:
            req = urllib.request.Request(ep, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": UA})
            with open_url(req, timeout=timeout + 10) as r:
                d = json.load(r)
            out = []
            for e in d.get("elements", []):
                lon = e.get("lon") or (e.get("center") or {}).get("lon")
                lat = e.get("lat") or (e.get("center") or {}).get("lat")
                if lon is None:
                    continue
                t = e.get("tags", {})
                out.append((lon, lat, t.get("name", ""), t.get("amenity") or t.get("healthcare") or ""))
            return out
        except Exception as exc:
            last = exc
    raise KitError(f"OpenStreetMap could not be reached ({last}).")


def roads(bbox, timeout=25):
    """Major roads (motorway to secondary) as lists of (lon, lat) vertices."""
    x0, y0, x1, y1 = bbox
    b = f"{y0},{x0},{y1},{x1}"
    q = (f"[out:json][timeout:{timeout}];way[\"highway\"~\"^(motorway|trunk|primary|secondary)(_link)?$\"]({b});"
         "out geom;")
    last = None
    for ep in ENDPOINTS:
        try:
            req = urllib.request.Request(ep, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": UA})
            with open_url(req, timeout=timeout + 10) as r:
                d = json.load(r)
            return [[(p["lon"], p["lat"]) for p in e.get("geometry", [])] for e in d.get("elements", []) if e.get("geometry")]
        except Exception as exc:
            last = exc
    raise KitError(f"OpenStreetMap could not be reached ({last}).")
