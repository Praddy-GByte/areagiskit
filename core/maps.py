# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Print-quality map layouts and charts (PNG bytes) with matplotlib.

A layout has: title, map with coordinate grid, north arrow, scale bar, legend,
location inset, projection and source notes. If matplotlib is not available,
callers fall back to plain previews.
"""

import io
import json
import math
import os

import numpy as np

from .raster import RAMPS, hex2rgb

try:
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    class _Plt:
        """The few pyplot calls used here, without pyplot: pyplot picks a GUI backend inside QGIS, and Qt figures
        created on a Processing worker thread can crash QGIS on Windows/macOS. Agg canvases are thread-safe and do not
        switch the backend that QGIS or other plugins use."""
        rcParams = matplotlib.rcParams

        @staticmethod
        def figure(figsize=None, dpi=None):
            f = Figure(figsize=figsize, dpi=dpi)
            FigureCanvasAgg(f)
            return f

        @staticmethod
        def subplots(figsize=None, dpi=None):
            f = _Plt.figure(figsize=figsize, dpi=dpi)
            return f, f.add_subplot(111)

        @staticmethod
        def close(fig=None):
            if fig is not None:
                fig.clear()

    plt = _Plt()
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import FancyArrow, Patch, Polygon, Rectangle
    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False

INK = "#1c2621"
MUTED = "#5d6b63"
RULE = "#c9cfc6"
FONT = {"family": "DejaVu Sans"}

_LAND = None


def _land():
    global _LAND
    if _LAND is None:
        p = os.path.join(os.path.dirname(__file__), "assets", "ne_110m_land.geojson")
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
            rings = []
            for f in d["features"]:
                g = f["geometry"]
                polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
                for poly in polys:
                    rings.append(np.array(poly[0]))
            _LAND = rings
        except Exception:
            _LAND = []
    return _LAND


class MapSpec:
    """What to draw. Either classes={value: (name, colour)} or ramp='name' with vmin/vmax."""

    def __init__(self, array, grid, title, classes=None, ramp=None, vmin=None, vmax=None, unit="",
                 subtitle="", source="", points=None, show_classes=None):
        self.array, self.grid, self.title = array, grid, title
        self.classes, self.ramp, self.vmin, self.vmax, self.unit = classes, ramp, vmin, vmax, unit
        self.subtitle, self.source, self.points = subtitle, source, points or []
        self.show_classes = show_classes


def _nice(x):
    if x <= 0:
        return 1
    e = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 2.5, 5, 10):
        if m * e >= x:
            return m * e
    return 10 * e


def _deg_step(span):
    for s in (0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10):
        if span / s <= 5:
            return s
    return 10


def _fmt_deg(v, axis):
    h = ("E" if v >= 0 else "W") if axis == "x" else ("N" if v >= 0 else "S")
    a = abs(v)
    d = int(a)
    m = (a - d) * 60
    if abs(m - round(m)) < 1e-6:
        return f"{d}°{int(round(m)):02d}′{h}"
    return f"{a:.3f}°{h}"


def _rgba_image(spec):
    a = spec.array
    g = spec.grid
    mask = g.mask
    if spec.classes:
        lut = np.zeros((256, 4), np.float32)
        for v, (_, col) in spec.classes.items():
            r, gg, b = hex2rgb(col)
            lut[int(v)] = (r / 255, gg / 255, b / 255, 1)
        img = lut[np.clip(np.nan_to_num(a, nan=0), 0, 255).astype(np.uint8)]
    else:
        cmap = LinearSegmentedColormap.from_list("r", RAMPS[spec.ramp])
        t = (np.nan_to_num(a, nan=spec.vmin) - spec.vmin) / ((spec.vmax - spec.vmin) or 1)
        img = cmap(np.clip(t, 0, 1)).astype(np.float32)
        img[~np.isfinite(a), 3] = 0
    img[~mask, 3] = 0
    return img


def _subsample(img, max_px=1600):
    s = max(1, int(math.ceil(max(img.shape[:2]) / max_px)))
    return img[::s, ::s]


def render_map(spec, width_in=7.0):
    """Return PNG bytes of a full map layout."""
    if not HAVE_MPL:
        return None
    g = spec.grid
    aoi = g.aoi
    x0, y0, x1, y1 = g.bounds
    mw, mh = x1 - x0, y1 - y0
    ratio = mh / mw
    map_w = width_in * 0.68
    map_h = min(max(map_w * ratio, 2.6), 6.2)
    fig_h = map_h + 1.15
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
    fig = plt.figure(figsize=(width_in, fig_h), dpi=220)
    fig.patch.set_facecolor("white")

    # title block
    fig.text(0.02, 1 - 0.18 / fig_h, spec.title, fontsize=11.5, fontweight="bold", color=INK, va="top")
    if spec.subtitle:
        fig.text(0.02, 1 - 0.43 / fig_h, spec.subtitle, fontsize=8, color=MUTED, va="top")

    top = 1 - 0.65 / fig_h
    bottom = 0.5 / fig_h
    ax = fig.add_axes([0.07, bottom, 0.66, top - bottom])
    img = _subsample(_rgba_image(spec))
    ax.imshow(img, extent=(x0, x1, y0, y1), origin="upper", interpolation="nearest", zorder=2)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f4f5f1")

    # area outline
    geom = aoi.local_geom
    polys = []
    gt = geom.GetGeometryType() % 1000
    if gt == 6:  # multipolygon
        polys = [geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())]
    elif gt == 3:
        polys = [geom]
    for p in polys:
        r = p.GetGeometryRef(0)
        pts = np.array(r.GetPoints())[:, :2]
        ax.plot(pts[:, 0], pts[:, 1], color=INK, lw=0.9, zorder=4)

    # graticule
    lx0, ly0, lx1, ly1 = aoi.bbox
    step = _deg_step(max(lx1 - lx0, ly1 - ly0))
    lons = np.arange(math.floor(lx0 / step) * step, lx1 + step, step)
    lats = np.arange(math.floor(ly0 / step) * step, ly1 + step, step)
    xt, xl, yt, yl = [], [], [], []
    for lon in lons:
        ll = np.linspace(ly0 - step, ly1 + step, 40)
        pts = np.array([aoi.to_local.TransformPoint(lon, la)[:2] for la in ll])
        ax.plot(pts[:, 0], pts[:, 1], color="#8a948d", lw=0.35, ls=(0, (3, 3)), zorder=3)
        xb = np.interp(y0, pts[:, 1], pts[:, 0])
        if x0 + mw * 0.04 < xb < x1 - mw * 0.04:
            xt.append(xb)
            xl.append(_fmt_deg(lon, "x"))
    for lat in lats:
        ll = np.linspace(lx0 - step, lx1 + step, 40)
        pts = np.array([aoi.to_local.TransformPoint(lo, lat)[:2] for lo in ll])
        ax.plot(pts[:, 0], pts[:, 1], color="#8a948d", lw=0.35, ls=(0, (3, 3)), zorder=3)
        yb = np.interp(x0, pts[:, 0], pts[:, 1])
        if y0 + mh * 0.04 < yb < y1 - mh * 0.04:
            yt.append(yb)
            yl.append(_fmt_deg(lat, "y"))
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=6.5, color=MUTED)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=6.5, color=MUTED, rotation=90, va="center")
    ax.tick_params(length=2.5, width=0.5, colors=MUTED, top=True, right=True, labeltop=False, labelright=False)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
        s.set_color(INK)

    # points (e.g. facilities, validation points)
    for (lon, lat), style in spec.points:
        x, y, _ = aoi.to_local.TransformPoint(lon, lat)
        ax.plot(x, y, marker=style.get("marker", "o"), ms=style.get("size", 3), color=style.get("color", INK),
                mec="white", mew=0.4, zorder=5, ls="none")

    # north arrow (map is centred on the area, so true north is up at the centre)
    nx, ny = x1 - mw * 0.06, y1 - mh * 0.16
    ah = mh * 0.08
    ax.add_patch(Rectangle((nx - ah * 0.55, ny - ah * 0.2), ah * 1.1, ah * 1.75, fc="white", ec="none", alpha=0.85, zorder=5))
    ax.add_patch(Polygon([[nx, ny + ah], [nx - ah * 0.32, ny], [nx, ny + ah * 0.28]], closed=True, fc=INK, ec=INK, lw=0.5, zorder=6))
    ax.add_patch(Polygon([[nx, ny + ah], [nx + ah * 0.32, ny], [nx, ny + ah * 0.28]], closed=True, fc="white", ec=INK, lw=0.5, zorder=6))
    ax.text(nx, ny + ah * 1.08, "N", ha="center", va="bottom", fontsize=8, fontweight="bold", color=INK, zorder=6,
            )

    # scale bar
    L = _nice(mw * 0.25)
    if L > mw * 0.4:
        L /= 2
    sx, sy = x0 + mw * 0.05, y0 + mh * 0.05
    sh = mh * 0.012
    ax.add_patch(Rectangle((sx - mw * 0.015, sy - sh * 2.2), L + mw * 0.06, sh * 6.2, fc="white", ec="none", alpha=0.85, zorder=6))
    for i in range(4):
        ax.add_patch(Rectangle((sx + i * L / 4, sy), L / 4, sh, fc=INK if i % 2 == 0 else "white", ec=INK, lw=0.5, zorder=7))
    unit, div = ("km", 1000) if L >= 1000 else ("m", 1)
    for v in (0, L / 2, L):
        lab = f"{v / div:g}"
        ax.text(sx + v, sy + sh * 1.6, lab, ha="center", va="bottom", fontsize=6.2, color=INK, zorder=7)
    ax.text(sx + L + mw * 0.012, sy + sh * 0.5, unit, ha="left", va="center", fontsize=6.2, color=INK, zorder=7)

    # legend
    lax = fig.add_axes([0.76, 0.42, 0.22, top - 0.42])
    lax.axis("off")
    lax.text(0, 1, "Legend", fontsize=8.5, fontweight="bold", color=INK, va="top", transform=lax.transAxes)
    if spec.classes:
        present = set(np.unique(spec.array[g.mask])) if spec.show_classes is None else set(spec.show_classes)
        items = [(n, c) for v, (n, c) in spec.classes.items() if v in present]
        y = 0.9
        dy = min(0.085, 0.86 / max(len(items), 1))
        for n, c in items:
            lax.add_patch(Rectangle((0, y - dy * 0.62), 0.11, dy * 0.6, fc=c, ec="#777", lw=0.4, transform=lax.transAxes))
            lax.text(0.15, y - dy * 0.32, n, fontsize=7, color=INK, va="center", transform=lax.transAxes)
            y -= dy
    else:
        cax = fig.add_axes([0.77, 0.47, 0.035, (top - 0.47) * 0.8])
        cmap = LinearSegmentedColormap.from_list("r", RAMPS[spec.ramp])
        cb = np.linspace(1, 0, 256)[:, None]
        cax.imshow(cb, aspect="auto", cmap=cmap, extent=(0, 1, spec.vmin, spec.vmax))
        cax.set_xticks([])
        cax.yaxis.tick_right()
        cax.tick_params(labelsize=6.5, length=2, colors=INK)
        cax.set_ylabel(spec.unit, fontsize=7, color=INK)
        cax.yaxis.set_label_position("right")
        for s in cax.spines.values():
            s.set_linewidth(0.5)

    # location inset
    iax = fig.add_axes([0.765, bottom, 0.215, 0.33])
    cx, cy = aoi.center
    span = 10
    iax.set_xlim(cx - span, cx + span)
    iax.set_ylim(cy - span * 0.8, cy + span * 0.8)
    iax.set_facecolor("#dfeaf2")
    for ring in _land():
        if ring[:, 0].max() < cx - span - 5 or ring[:, 0].min() > cx + span + 5:
            continue
        iax.fill(ring[:, 0], ring[:, 1], fc="#f2efe6", ec="#9aa39c", lw=0.3)
    bx0, by0, bx1, by1 = aoi.bbox
    w, h = max(bx1 - bx0, 0.35), max(by1 - by0, 0.35)
    iax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="#c8281e", ec="#c8281e", lw=1))
    iax.set_xticks([])
    iax.set_yticks([])
    for s in iax.spines.values():
        s.set_linewidth(0.6)
        s.set_color(INK)
    iax.set_title("Location", fontsize=7, color=INK, pad=2, loc="left")

    # notes
    note = (f"Projection: Lambert Azimuthal Equal Area centred at {abs(cy):.3f}°{'N' if cy >= 0 else 'S'} "
            f"{abs(cx):.3f}°{'E' if cx >= 0 else 'W'} · Pixel {g.res:.0f} m")
    fig.text(0.07, 0.17 / fig_h, note, fontsize=6, color=MUTED)
    if spec.source:
        fig.text(0.07, 0.04 / fig_h, "Data: " + spec.source, fontsize=6, color=MUTED)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _chart_fig(width_in, height_in):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=220)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=7.5, length=2.5, width=0.6)
    return fig, ax


def _png(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=220, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def render_bars(rows, xlabel="", signed=False, width_in=6.3):
    """Horizontal bars. rows: (label, value, colour)."""
    if not HAVE_MPL or not rows:
        return None
    h = 0.42 * len(rows) + 0.8
    fig, ax = _chart_fig(width_in, h)
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [r[2] for r in rows][::-1]
    ax.barh(labels, vals, color=cols, edgecolor="#555", linewidth=0.4, height=0.62)
    m = max(abs(v) for v in vals) or 1
    for i, v in enumerate(vals):
        ax.text(v + (m * 0.015 if v >= 0 else -m * 0.015), i, f"{v:+,.2f}" if signed else f"{v:,.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7, color=INK)
    if signed:
        ax.axvline(0, color=INK, lw=0.7)
        ax.set_xlim(-m * 1.25, m * 1.25)
    else:
        ax.set_xlim(0, m * 1.2)
    ax.grid(axis="x", color=RULE, lw=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=7.5, color=MUTED)
    return _png(fig)


def render_line(series, xlabels, ylabel="", ymin=None, ymax=None, width_in=6.3, height_in=2.8):
    if not HAVE_MPL:
        return None
    fig, ax = _chart_fig(width_in, height_in)
    x = np.arange(len(xlabels))
    for name, vals, col in series:
        v = np.array([np.nan if a is None else a for a in vals], float)
        ax.plot(x, v, color=col, lw=1.8, marker="o", ms=3, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=0, fontsize=6.8)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel, fontsize=7.5, color=MUTED)
    ax.grid(axis="y", color=RULE, lw=0.4)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=7)
    return _png(fig)


def render_polar(series, labels, title_unit="km²", width_in=4.6):
    """Polar bar chart, e.g. new built-up area by compass sector. series: [(name, values, colour)]."""
    if not HAVE_MPL:
        return None
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig = plt.figure(figsize=(width_in, width_in), dpi=220)
    ax = fig.add_subplot(111, projection="polar")
    n = len(labels)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    w = 2 * np.pi / n / max(len(series), 1) * 0.9
    for i, (name, vals, col) in enumerate(series):
        ax.bar(theta + (i - (len(series) - 1) / 2) * w, vals, width=w, color=col, alpha=0.9, label=name, edgecolor="white", linewidth=0.4)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=8, color=INK)
    ax.tick_params(axis="y", labelsize=6.5, colors=MUTED)
    ax.set_title(title_unit, fontsize=7.5, color=MUTED, loc="left")
    if len(series) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=7, frameon=False)
    return _png(fig)


def render_multi_line(series, x, xlabel="", ylabel="", width_in=6.3, height_in=2.8):
    """Lines over numeric x. series: [(name, values, colour)]."""
    if not HAVE_MPL:
        return None
    fig, ax = _chart_fig(width_in, height_in)
    for name, vals, col in series:
        v = np.array([np.nan if a is None else a for a in vals], float)
        ax.plot(x[:len(v)], v, color=col, lw=1.8, marker="o", ms=3, label=name)
    ax.set_xlabel(xlabel, fontsize=7.5, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=7.5, color=MUTED)
    ax.grid(color=RULE, lw=0.4)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    return _png(fig)


def render_grouped_bars(groups, series, ylabel="", width_in=6.3, height_in=3.0, ref_lines=None):
    """Vertical grouped bars. groups: labels; series: [(name, values, colour)]; ref_lines: [(y, label)]."""
    if not HAVE_MPL:
        return None
    fig, ax = _chart_fig(width_in, height_in)
    x = np.arange(len(groups))
    w = 0.8 / max(len(series), 1)
    for i, (name, vals, col) in enumerate(series):
        ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, width=w, color=col, label=name, edgecolor="#555", linewidth=0.3)
    for y, lab in (ref_lines or []):
        ax.axhline(y, color=INK, lw=0.9, ls="--")
        ax.text(len(groups) - 0.5, y, " " + lab, va="bottom", ha="right", fontsize=6.8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=7, rotation=0)
    ax.set_ylabel(ylabel, fontsize=7.5, color=MUTED)
    ax.grid(axis="y", color=RULE, lw=0.4)
    ax.set_axisbelow(True)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=7, ncol=min(len(series), 4))
    return _png(fig)
