# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Reports as a structured chapter, written to HTML, Word (.docx) and PDF.

Every kit fills the same chapter:
  Summary (one page) → 1 Study area → 2 Data → 3 Method → 4 Results →
  5 Validation → 6 Limitations → 7 Conclusions → References →
  Appendix A Proof (every input scene, linked; rerun command) → Appendix B Files

Figures and tables are numbered automatically. Text may contain <b>…</b> only.
Word files are written with the standard library (no python-docx needed);
PDF uses Qt, which ships with QGIS.
"""

import base64
import html
import io
import os
import re
import struct
import time
import zipfile

from .maps import HAVE_MPL, MapSpec, render_bars, render_line, render_map
from .raster import preview_classes, preview_float
from .net import skipped

E = html.escape


PROOF_MAX_ROWS = 80   # longer scene lists are summarised in the report and written in full to proof_scenes.csv


def fmt(v, d=2):
    try:
        return f"{v:,.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def grade(checks):
    """checks: list of (points, max_points, reason). Returns {'level', 'reasons', 'score'}."""
    got = sum(c[0] for c in checks)
    mx = sum(c[1] for c in checks) or 1
    frac = got / mx
    level = "High" if frac >= 0.75 else "Medium" if frac >= 0.45 else "Low"
    return {"level": level, "reasons": [c[2] for c in checks], "score": round(frac * 100)}


def _bold_html(t):
    """Escape text but keep <b> tags."""
    t = E(str(t))
    return t.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")


def _plain(t):
    return re.sub(r"</?b>", "", str(t))


def _png_size(data):
    w, h = struct.unpack(">II", data[16:24])
    return w, h


# Inside QGIS the PDF is written on the main thread (Qt text layout), so the plugin sets this
# and calls write_deferred_pdf() from postProcessAlgorithm.
DEFER_PDF = False

LEVEL_COL = {"High": "#2f6b4f", "Medium": "#a5700f", "Low": "#b23a2b"}


# "What this report will not tell you": shown on the summary page of every report, so limits are read before results.
WONT = {
    "Flood": [
        "The exact water depth at a house: depth comes from a 30 m elevation model whose vertical error (2–4 m) can be larger than the water depth.",
        "A safe evacuation route: \"safe high ground\" is a straight-line distance and direction, not a checked route. Follow official evacuation plans.",
        "When the next flood will come: the river-flow forecast is the GloFAS model at one river cell with no local skill score; after 2–4 weeks it is only an outlook.",
        "Losses you can rely on: damage curves are continental averages (JRC), not calibrated for this area; money appears only if you give asset values.",
        "Engineering flood lines: 10/100-year zones come from global models and a simple terrain model, for screening only.",
        "Water the radar cannot see: under dense trees, between tall buildings, and very short floods between satellite passes.",
        "An independent accuracy for susceptibility: the hold-out AUC is optimistic because the flood inventory comes from the same satellite record.",
    ],
    "Landslide": [
        "Whether a particular slope will fail: the factor of safety uses textbook soil strengths from global soil maps, not site tests. \"Unstable\" means relatively weakest, not \"will fall\".",
        "When the next landslide will happen: timing needs rainfall warnings. The 1/10/50-year chances come from only a few years of satellite record.",
        "Exact debris reach and speed: runout uses general parameters (11° travel angle, speed capped at 15 m/s), not values calibrated for this region.",
        "Landslides smaller than about 20 m, under forest canopy, or slow-moving ground (that needs InSAR, which needs an account).",
        "The rain that triggered a landslide on a slope: rainfall comes from ~50 km cells and can miss mountain downpours by several times (Wayanad 2024: 88 mm in the cell vs 573 mm at a gauge 5 km away).",
        "An independent accuracy, unless you supply an inventory: automatic AUCs are optimistic because training and test scars come from the same detection.",
    ],
    "LULC": [
        "An independent accuracy, unless you label the accuracy sample: automatic labels come from Esri and ESA WorldCover, and the hold-out test uses the same labels.",
        "Measured carbon, ecosystem value or runoff: these use transferred coefficients (IPCC Tier 1, Costanza 1997, US curve numbers).",
        "The future: projections are produced only if the method reproduces the recent past, and even then they are scenarios, not forecasts.",
        "Detailed classes (crop types, plantations, mining) unless you give your own training points.",
    ],
    "Crop": [
        "Yield in tonnes: MODIS GPP tracks good and bad years but is plant carbon uptake, not harvested grain, and it understates the size of losses.",
        "Crop names, unless you give field points with crop labels.",
        "A tested irrigation map for this area: irrigated vs rainfed was checked in one region (California) and is indicative elsewhere.",
        "Field-level rain or soil moisture: these come from ~50 km climate cells.",
    ],
    "Urban": [
        "Air temperature in a street: hourly air temperature, heat index, WBGT and UTCI come from ~50 km NASA POWER cells, so the whole city gets one or two values.",
        "Heat stress in shade: UTCI uses a simple open-ground radiation model; shaded streets are not modelled. Surface temperature is not air temperature.",
        "Ground-level pollution: NO₂ is a column through the atmosphere, not what people breathe; PM2.5 is the CAMS model at ~40 km and often reads low in South Asia.",
        "Individual trees or buildings: the data are 10–100 m.",
    ],
    "Health": [
        "Causes: links with environment factors are associations, not proof of cause.",
        "An official outbreak alert: the early warning has been tested on simulations, not on real outbreaks.",
        "Individual risk: every number is for an area, not a person.",
        "Anything about consent: you are responsible for permission to use the case data. Small counts are hidden by default.",
    ],
}


def wont_for(kit_name):
    for k, v in WONT.items():
        if k.lower() in (kit_name or "").lower():
            return v
    return []


class Report:
    def __init__(self, kit, title, place, verdict=None, accent="#2f6b4f", aoi=None, grid=None):
        self.kit, self.title, self.place, self.verdict, self.accent = kit, title, place, verdict, accent
        self.aoi, self.grid = aoi, grid
        self.key_results, self.key_points, self.key_figure = [], [], None
        self.area_facts, self.area_figure = [], None
        self.data_rows, self.method_steps, self.results = [], [], []
        self.checks, self.limit_items, self.conclusion_items = [], [], []
        self.refs, self.proof_sources, self.command, self.file_list = [], [], "", []
        self.proof_csv_rows = None
        self.run_facts = []

    # ------------------------------------------------------------------ content
    def summary(self, results, points, figure=None):
        """results: [(label, value)], points: short sentences, figure: MapSpec for the summary page."""
        self.key_results, self.key_points, self.key_figure = results, points, figure

    def study_area(self, figure=None, extra_facts=()):
        facts = []
        if self.aoi is not None:
            a = self.aoi
            x0, y0, x1, y1 = a.bbox
            facts += [("Name", a.name), ("Area", f"{fmt(a.area_km2)} km²"),
                      ("Centre", f"{abs(a.center[1]):.4f}°{'N' if a.center[1] >= 0 else 'S'}, "
                                 f"{abs(a.center[0]):.4f}°{'E' if a.center[0] >= 0 else 'W'}"),
                      ("Extent", f"{x0:.4f}°–{x1:.4f}° E, {y0:.4f}°–{y1:.4f}° N")]
        if self.grid is not None:
            facts += [("Analysis grid", f"{self.grid.res:.0f} m pixels, {self.grid.cols:,} × {self.grid.rows:,}"),
                      ("Projection", "Lambert Azimuthal Equal Area centred on the area (true areas in km²)")]
        facts += list(extra_facts)
        self.area_facts, self.area_figure = facts, figure

    def data(self, rows):
        """rows: (dataset, what it is used for, resolution, dates used, licence)."""
        self.data_rows = rows

    def method(self, steps):
        self.method_steps = steps

    def heading(self, text):
        self.results.append(("h2", text))

    def text(self, t):
        self.results.append(("p", t))

    def bullets(self, items):
        self.results.append(("ul", items))

    def map(self, spec, caption):
        self.results.append(("map", spec, caption))

    def bars(self, rows, caption, xlabel="", signed=False):
        self.results.append(("bars", rows, caption, xlabel, signed))

    def line(self, series, xlabels, caption, ylabel="", ymin=None, ymax=None):
        self.results.append(("line", series, xlabels, caption, ylabel, ymin, ymax))

    def figure(self, png, caption):
        """Any ready-made PNG (for example a chart from core.maps)."""
        self.results.append(("png", png, caption))

    def table(self, headers, rows, caption, swatches=None):
        self.results.append(("table", headers, rows, caption, swatches))

    def validation(self, rows):
        """rows: (check, result, what it means)."""
        self.checks = rows

    def limits(self, items):
        self.limit_items = items

    def conclusions(self, items):
        self.conclusion_items = items

    def references(self, items):
        self.refs = items

    def proof(self, sources, command):
        self.proof_sources, self.command = sources, command

    def files(self, paths):
        self.file_list = [os.path.basename(p) for p in paths]

    # ------------------------------------------------------------------ assembly
    def _figure_png(self, spec):
        if isinstance(spec, MapSpec):
            png = render_map(spec) if HAVE_MPL else None
            if png is None:  # plain preview fallback
                if spec.classes:
                    png = preview_classes(spec.array, spec.classes)[0]
                else:
                    png = preview_float(spec.array, spec.ramp, spec.vmin, spec.vmax, spec.grid.mask)[0]
            return png
        return spec

    def _blocks(self, seconds):
        """Flatten the chapter into a list of blocks with numbered figures and tables."""
        B = []
        fig_n = [0]
        tab_n = [0]

        def fig(png, caption, small=False):
            if png is None:
                return
            fig_n[0] += 1
            B.append(("fig", png, f"Figure {fig_n[0]}. {caption}", small))

        def tab(headers, rows, caption, sw=None):
            tab_n[0] += 1
            B.append(("table", headers, rows, f"Table {tab_n[0]}. {caption}", sw))

        # summary page
        B.append(("title", self.title, f"{self.kit} · {self.place}"))
        meta = [time.strftime("%d %B %Y")]
        if seconds is not None:
            meta.append(f"Run time {seconds:.0f} s")
        B.append(("meta", " · ".join(meta)))
        if self.verdict:
            B.append(("grade", self.verdict["level"], self.verdict["score"]))
        B.append(("h1x", "Summary"))
        if self.key_results:
            B.append(("kv", self.key_results))
        if self.key_points:
            B.append(("ul", self.key_points))
        wn = wont_for(self.kit)
        if wn:
            B.append(("h2", "What this report will not tell you"))
            B.append(("ul", wn))
        if self.key_figure is not None:
            fig(self._figure_png(self.key_figure), self.key_figure.title, small=True)
        B.append(("pagebreak",))

        sec = [0]

        def h1(t):
            sec[0] += 1
            B.append(("h1", f"{sec[0]}. {t}"))

        h1("Study area")
        if self.area_facts:
            tab(["Item", "Value"], [[k, v] for k, v in self.area_facts], "Study area at a glance")
        if self.area_figure is not None:
            fig(self._figure_png(self.area_figure), self.area_figure.title)

        h1("Data")
        if self.data_rows:
            tab(["Dataset", "Used for", "Resolution", "Dates used", "Licence"], [list(r) for r in self.data_rows],
                "Input data. Every scene is listed and linked in Appendix A")

        h1("Method")
        if self.method_steps:
            B.append(("steps", self.method_steps))

        h1("Results")
        for b in self.results:
            k = b[0]
            if k == "h2":
                B.append(("h2", b[1]))
            elif k == "p":
                B.append(("p", b[1]))
            elif k == "ul":
                B.append(("ul", b[1]))
            elif k == "map":
                fig(self._figure_png(b[1]), b[2])
            elif k == "bars":
                fig(render_bars(b[1], b[3], b[4]), b[2])
            elif k == "line":
                fig(render_line(b[1], b[2], b[4], b[5], b[6]), b[3])
            elif k == "table":
                tab(b[1], b[2], b[3], b[4])
            elif k == "png":
                fig(b[1], b[2])

        h1("Validation")
        if self.verdict:
            B.append(("p", f"Overall confidence: <b>{self.verdict['level']}</b> (score {self.verdict['score']} of 100). "
                           "The grade combines the checks below; each check is scored and none is entered by hand."))
        if self.checks:
            tab(["Check", "Result", "What it means"], [list(r) for r in self.checks], "Validation checks")

        h1("Limitations")
        B.append(("ul", self.limit_items))

        h1("Conclusions")
        B.append(("ul", self.conclusion_items))

        B.append(("h1", "References"))
        B.append(("ol", self.refs))

        B.append(("pagebreak",))
        B.append(("h1", "Appendix A. Proof: data used and how to reproduce"))
        B.append(("p", "Nothing in this report is typed in by hand: every number is computed from the files listed here. "
                       "Each record link opens the original public catalogue entry, so anyone can check the date, "
                       "cloud cover and footprint of the exact scene used."))
        if self.proof_sources:
            src = list(self.proof_sources)
            if len(src) <= PROOF_MAX_ROWS:
                tab(["Dataset", "Scene / tile", "Date", "Record"], [[c, i, d, u] for c, i, d, u in src],
                    "Every input scene and tile read for this report")
            else:
                # long lists go to proof_scenes.csv (written next to the report); the report keeps a summary per dataset
                self.proof_csv_rows = src
                by = {}
                for c, i, d, u in src:
                    s = by.setdefault(c, [0, None, None, u])
                    s[0] += 1
                    ds = str(d or "")[:10]
                    if ds:
                        s[1] = ds if s[1] is None or ds < s[1] else s[1]
                        s[2] = ds if s[2] is None or ds > s[2] else s[2]
                tab(["Dataset", "Scenes / tiles", "First date", "Last date", "Example record"],
                    [[c, f"{n:,}", a or "–", b or "–", u] for c, (n, a, b, u) in sorted(by.items(), key=lambda kv: -kv[1][0])],
                    f"Input data by dataset ({len(src):,} scenes and tiles; every one is listed with its link in proof_scenes.csv)")
        if self.command:
            B.append(("p", "<b>Rerun it yourself.</b> This command repeats the analysis on any computer with Python and GDAL "
                           "(for example the QGIS Python console) and gives the same numbers, unless the data provider has "
                           "reprocessed these scenes since."))
            B.append(("code", self.command))
        if self.file_list:
            B.append(("h1", "Appendix B. Files"))
            B.append(("ul", self.file_list))
        return B

    # ------------------------------------------------------------------ writers
    def write(self, out_dir, seconds=None, basename="report"):
        B = self._blocks(seconds)
        out = {}
        if self.proof_csv_rows:
            import csv
            pc = os.path.join(out_dir, "proof_scenes.csv")
            with open(pc, "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["dataset", "scene_or_tile", "date", "record_url"])
                w.writerows(self.proof_csv_rows)
            out_csv = pc
        else:
            out_csv = None
        out["html"] = os.path.join(out_dir, basename + ".html")
        with open(out["html"], "w", encoding="utf-8") as fh:
            fh.write(self._html(B))
        out["docx"] = os.path.join(out_dir, basename + ".docx")
        _write_docx(B, out["docx"], self.title, self.accent)
        pdf = os.path.join(out_dir, basename + ".pdf")
        if DEFER_PDF:
            out["pdf"] = pdf
            out["pdf_job"] = (B, pdf, self.accent)
        elif _write_pdf(B, pdf, self.accent):
            out["pdf"] = pdf
        return out


    def _html(self, B):
        parts = []
        for b in B:
            k = b[0]
            if k == "title":
                parts.append(f'<header><span class="eyebrow">{E(b[2])}</span><h1 class="t">{E(b[1])}</h1>')
            elif k == "meta":
                parts.append(f'<div class="meta">{E(b[1])}</div>')
            elif k == "grade":
                parts.append(f'<span class="pill" style="color:{LEVEL_COL[b[1]]}"><i></i>Confidence: {b[1]} · {b[2]}/100</span></header>')
            elif k in ("h1", "h1x"):
                parts.append(f"<h2>{E(b[1])}</h2>")
            elif k == "h2":
                parts.append(f"<h3>{E(b[1])}</h3>")
            elif k == "p":
                parts.append(f"<p>{_bold_html(b[1])}</p>")
            elif k == "ul":
                parts.append('<ul>' + "".join(f"<li>{_bold_html(x)}</li>" for x in b[1]) + "</ul>")
            elif k == "ol":
                parts.append('<ol class="refs">' + "".join(f"<li>{_bold_html(x)}</li>" for x in b[1]) + "</ol>")
            elif k == "steps":
                parts.append('<ol class="steps">' + "".join(f"<li><b>{E(t)}.</b> {_bold_html(x)}</li>" for t, x in b[1]) + "</ol>")
            elif k == "kv":
                parts.append('<div class="kv">' + "".join(
                    f'<div><span>{E(a)}</span><b>{_bold_html(v)}</b></div>' for a, v in b[1]) + "</div>")
            elif k == "fig":
                parts.append(f'<figure{" class=sm" if len(b) > 3 and b[3] else ""}><img src="data:image/png;base64,{base64.b64encode(b[1]).decode()}" alt="">'
                             f"<figcaption>{E(b[2])}</figcaption></figure>")
            elif k == "table":
                _, headers, rows, cap, sw = b
                head = "".join(f"<th>{E(h)}</th>" for h in headers)
                body = []
                for i, r in enumerate(rows):
                    cells = []
                    for j, c in enumerate(r):
                        c = str(c)
                        if c.startswith("http"):
                            cells.append(f'<td><a href="{E(c)}" target="_blank" rel="noopener">open</a></td>')
                            continue
                        dot = (f'<span class="sw" style="background:{sw[i]}"></span>' if (sw and j == 0 and i < len(sw) and sw[i]) else "")
                        cells.append(f"<td>{dot}{_bold_html(c)}</td>")
                    body.append("<tr>" + "".join(cells) + "</tr>")
                parts.append(f'<div class="tw"><div class="cap">{E(cap)}</div><div class="scroll"><table><thead><tr>{head}</tr></thead>'
                             f'<tbody>{"".join(body)}</tbody></table></div></div>')
            elif k == "code":
                parts.append(f'<pre class="cmd">{E(b[1])}</pre>')
            elif k == "pagebreak":
                parts.append('<hr class="pb">')
        return HTML_TMPL.format(title=E(self.title), accent=self.accent, body="\n".join(parts))


HTML_TMPL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --paper:#f6f7f4; --card:#fff; --ink:#1c2621; --muted:#5d6b63; --rule:#dde2da; --accent:{accent}; --tint:#eef1ec }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --paper:#151a17; --card:#1d2420; --ink:#e6ebe7; --muted:#9ba89f; --rule:#2e3833; --tint:#222a25; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --paper:#151a17; --card:#1d2420; --ink:#e6ebe7; --muted:#9ba89f; --rule:#2e3833; --tint:#222a25; color-scheme:dark }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.6 "Segoe UI", system-ui, -apple-system, Roboto, sans-serif;
  padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); padding-block:28px 64px }}
main {{ max-width:860px; margin:0 auto; background:var(--card); border:1px solid var(--rule); padding:clamp(18px,4vw,48px); display:flex; flex-direction:column; gap:14px }}
header {{ display:flex; flex-direction:column; gap:8px; border-bottom:3px solid var(--accent); padding-bottom:16px }}
.eyebrow {{ font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); font-weight:600 }}
h1.t {{ font:700 clamp(24px,3.6vw,34px)/1.15 Georgia, "Times New Roman", serif; margin:0; text-wrap:balance }}
.meta {{ color:var(--muted); font-size:13px }}
.pill {{ display:inline-flex; gap:8px; align-items:center; border:1.5px solid currentColor; border-radius:99px; padding:3px 12px; font-weight:600; font-size:13px; width:fit-content }}
.pill i {{ width:9px; height:9px; border-radius:50%; background:currentColor }}
h2 {{ font:700 21px/1.25 Georgia, serif; margin:26px 0 2px; color:var(--ink); border-bottom:1px solid var(--rule); padding-bottom:6px }}
h3 {{ font-size:16px; margin:14px 0 0 }}
p, li {{ max-width:72ch }}
p {{ margin:0 }}
ul, ol {{ margin:0; padding-left:22px; display:flex; flex-direction:column; gap:5px }}
ol.steps li {{ padding-left:4px }}
ol.refs {{ font-size:13px; color:var(--muted) }}
.kv {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(170px,1fr)); gap:1px; background:var(--rule); border:1px solid var(--rule) }}
.kv div {{ background:var(--card); padding:12px 14px; display:flex; flex-direction:column; gap:2px }}
.kv span {{ font-size:12px; color:var(--muted) }} .kv b {{ font-size:19px; font-variant-numeric:tabular-nums }}
figure {{ margin:8px 0; display:flex; flex-direction:column; gap:6px }}
figure.sm img {{ max-width:620px }}
figure img {{ width:100%; height:auto; border:1px solid var(--rule); background:#fff }}
figcaption, .cap {{ font-size:13px; color:var(--muted); font-style:italic }}
.tw {{ display:flex; flex-direction:column; gap:6px; margin:6px 0 }}
.scroll {{ overflow-x:auto }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px }}
th {{ text-align:left; background:var(--tint); font-weight:600; padding:7px 10px; border-bottom:2px solid var(--rule) }}
td {{ padding:6px 10px; border-bottom:1px solid var(--rule); vertical-align:top; font-variant-numeric:tabular-nums }}
td:first-child {{ white-space:nowrap }}
.sw {{ display:inline-block; width:11px; height:11px; border-radius:2px; margin-right:7px; vertical-align:-1px; border:1px solid rgba(0,0,0,.2) }}
pre.cmd {{ background:var(--tint); border:1px solid var(--rule); padding:12px; font:12px/1.5 ui-monospace, Consolas, monospace; white-space:pre-wrap; word-break:break-all; margin:0 }}
a {{ color:var(--accent) }}
hr.pb {{ border:none; border-top:1px dashed var(--rule); margin:22px 0 }}
@media print {{ body {{ background:#fff; padding:0 }} main {{ border:none }} hr.pb {{ page-break-after:always; border:none }} }}
</style></head>
<body><main>
{body}
</main></body></html>"""


# ---------------------------------------------------------------------- Word

def _x(t):
    return html.escape(str(t), quote=True)


def _runs(text, size=None, color=None, italic=False, bold_all=False):
    out = []
    bold = False
    for part in re.split(r"(</?b>)", str(text)):
        if part == "<b>":
            bold = True
            continue
        if part == "</b>":
            bold = False
            continue
        if not part:
            continue
        rpr = ""
        if bold or bold_all:
            rpr += "<w:b/>"
        if italic:
            rpr += "<w:i/>"
        if color:
            rpr += f'<w:color w:val="{color.lstrip("#")}"/>'
        if size:
            rpr += f'<w:sz w:val="{int(size * 2)}"/>'
        out.append(f'<w:r>{"<w:rPr>" + rpr + "</w:rPr>" if rpr else ""}<w:t xml:space="preserve">{_x(part)}</w:t></w:r>')
    return "".join(out)


def _p(text, style=None, align=None, size=None, color=None, italic=False, bold=False, ind=None, keep=False, after=None):
    ppr = ""
    if style:
        ppr += f'<w:pStyle w:val="{style}"/>'
    if keep:
        ppr += "<w:keepNext/>"
    if after is not None:
        ppr += f'<w:spacing w:after="{after}"/>'
    if ind:
        ppr += f'<w:ind w:left="{ind}" w:hanging="260"/>'
    if align:
        ppr += f'<w:jc w:val="{align}"/>'
    return f'<w:p>{"<w:pPr>" + ppr + "</w:pPr>" if ppr else ""}{_runs(text, size, color, italic, bold)}</w:p>'


def _cell(content_xml, width, fill=None, header=False):
    tcpr = f'<w:tcW w:w="{width}" w:type="dxa"/>'
    if fill:
        tcpr += f'<w:shd w:val="clear" w:color="auto" w:fill="{fill.lstrip("#")}"/>'
    return f"<w:tc><w:tcPr>{tcpr}</w:tcPr>{content_xml}</w:tc>"


def _write_docx(B, path, title, accent):
    media, rels = [], []
    body = []
    pic_id = [0]

    def image(png, width_cm=16.0, max_h_cm=21.5):
        pic_id[0] += 1
        n = pic_id[0]
        rid = f"rIdImg{n}"
        media.append((f"media/image{n}.png", png))
        rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image{n}.png"/>')
        w, h = _png_size(png)
        cx = int(width_cm / 2.54 * 914400)
        cy = int(cx * h / w)
        max_cy = int((11.5 if width_cm < 16 else max_h_cm) / 2.54 * 914400)
        if cy > max_cy:
            cx, cy = int(cx * max_cy / cy), max_cy
        return (f'<w:p><w:pPr><w:keepNext/><w:jc w:val="center"/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
                f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{n}" name="Figure {n}"/>'
                '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                f'<pic:nvPicPr><pic:cNvPr id="{n}" name="image{n}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
                f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
                f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
                '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')

    def table(headers, rows, sw=None, widths=None):
        total = 9638
        n = len(headers)
        widths = widths or [total // n] * n
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        trs = ['<w:tr><w:trPr><w:tblHeader/></w:trPr>' + "".join(
            _cell(_p(h, size=9, bold=True, after=0), widths[j], "E7ECE5") for j, h in enumerate(headers)) + "</w:tr>"]
        for i, r in enumerate(rows):
            cells = []
            for j, c in enumerate(r):
                c = str(c)
                txt = _runs(c, 9)
                if sw and j == 0 and i < len(sw) and sw[i]:
                    txt = f'<w:r><w:rPr><w:color w:val="{sw[i].lstrip("#")}"/><w:sz w:val="20"/></w:rPr><w:t xml:space="preserve">■ </w:t></w:r>' + txt
                cells.append(_cell(f'<w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>{txt}</w:p>', widths[j]))
            trs.append("<w:tr><w:trPr><w:cantSplit/></w:trPr>" + "".join(cells) + "</w:tr>")
        return ('<w:tbl><w:tblPr><w:tblStyle w:val="GKTable"/><w:tblW w:w="5000" w:type="pct"/>'
                '<w:tblLayout w:type="autofit"/></w:tblPr>'
                f"<w:tblGrid>{grid}</w:tblGrid>{''.join(trs)}</w:tbl>" + _p("", after=120))

    acc = accent.lstrip("#")
    for b in B:
        k = b[0]
        if k == "title":
            body.append(_p(b[2].upper(), size=9, color=acc, bold=True))
            body.append(_p(b[1], style="Title"))
        elif k == "meta":
            body.append(_p(b[1], size=9.5, color="5D6B63"))
        elif k == "grade":
            body.append(_p(f"Confidence: {b[1]} ({b[2]}/100)", size=11, bold=True, color=LEVEL_COL[b[1]].lstrip("#")))
        elif k in ("h1", "h1x"):
            body.append(_p(b[1], style="Heading1", keep=True))
        elif k == "h2":
            body.append(_p(b[1], style="Heading2", keep=True))
        elif k == "p":
            body.append(_p(b[1]))
        elif k in ("ul", "ol"):
            for i, x in enumerate(b[1]):
                body.append(_p(("• " if k == "ul" else f"{i + 1}. ") + str(x), ind=360, after=60))
        elif k == "steps":
            for i, (t, x) in enumerate(b[1]):
                body.append(_p(f"{i + 1}. <b>{t}.</b> {x}", ind=360, after=80))
        elif k == "kv":
            rows = [[a, v] for a, v in b[1]]
            body.append(table(["Key result", "Value"], rows, widths=[4800, 4838]))
        elif k == "fig":
            body.append(image(b[1], 13.0 if len(b) > 3 and b[3] else 16.0))
            body.append(_p(b[2], style="Caption"))
        elif k == "table":
            _, headers, rows, cap, sw = b
            body.append(_p(cap, style="Caption", keep=True))
            body.append(table(headers, rows, sw))
        elif k == "code":
            body.append(f'<w:p><w:pPr><w:shd w:val="clear" w:color="auto" w:fill="EEF1EC"/></w:pPr>'
                        f'<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="16"/></w:rPr>'
                        f'<w:t xml:space="preserve">{_x(b[1])}</w:t></w:r></w:p>')
        elif k == "pagebreak":
            body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    sect = ('<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter"/>'
            '<w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" '
            'w:header="567" w:footer="567" w:gutter="0"/></w:sectPr>')
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
           'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
           'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
           'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
           'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
           f"<w:body>{''.join(body)}{sect}</w:body></w:document>")
    footer = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
              'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
              f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr>{_runs(_plain(title) + "  ·  page ", 8, "8A948D")}'
              '<w:r><w:rPr><w:sz w:val="16"/><w:color w:val="8A948D"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>'
              '<w:r><w:rPr><w:sz w:val="16"/><w:color w:val="8A948D"/></w:rPr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
              '<w:r><w:rPr><w:sz w:val="16"/><w:color w:val="8A948D"/></w:rPr><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')
    styles = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/><w:sz w:val="21"/><w:szCs w:val="21"/><w:lang w:val="en-GB"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/>
<w:pPr><w:spacing w:after="80"/></w:pPr><w:rPr><w:rFonts w:ascii="Georgia" w:hAnsi="Georgia"/><w:b/><w:sz w:val="44"/><w:color w:val="1C2621"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
<w:pPr><w:keepNext/><w:spacing w:before="360" w:after="120"/><w:pBdr><w:bottom w:val="single" w:sz="6" w:space="2" w:color="{acc}"/></w:pBdr><w:outlineLvl w:val="0"/></w:pPr>
<w:rPr><w:rFonts w:ascii="Georgia" w:hAnsi="Georgia"/><w:b/><w:sz w:val="30"/><w:color w:val="1C2621"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
<w:pPr><w:keepNext/><w:spacing w:before="240" w:after="80"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="{acc}"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="caption"/><w:basedOn w:val="Normal"/><w:qFormat/>
<w:pPr><w:spacing w:before="60" w:after="200"/></w:pPr><w:rPr><w:i/><w:sz w:val="18"/><w:color w:val="5D6B63"/></w:rPr></w:style>
<w:style w:type="table" w:styleId="GKTable"><w:name w:val="AreaGIS Kit Table"/><w:tblPr>
<w:tblBorders><w:top w:val="single" w:sz="8" w:color="1C2621"/><w:bottom w:val="single" w:sz="8" w:color="1C2621"/><w:insideH w:val="single" w:sz="4" w:color="D5DBD2"/></w:tblBorders>
<w:tblCellMar><w:top w:w="40" w:type="dxa"/><w:left w:w="90" w:type="dxa"/><w:bottom w:w="40" w:type="dxa"/><w:right w:w="90" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>
</w:styles>"""
    ctypes = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
              '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
              '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
              '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
              '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
              '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
              '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/></Types>')
    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                 '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                 "</Relationships>")
    doc_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                '<Relationship Id="rIdFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>'
                + "".join(rels) + "</Relationships>")
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f"<dc:title>{_x(_plain(title))}</dc:title><dc:creator>{_x(_creator())}</dc:creator>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created></cp:coreProperties>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ctypes)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("word/document.xml", doc)
        z.writestr("word/styles.xml", styles)
        z.writestr("word/footer1.xml", footer)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("docProps/core.xml", core)
        for name, data in media:
            z.writestr("word/" + name, data)
    return path


# ---------------------------------------------------------------------- PDF (Qt)

def _write_pdf(B, path, accent):
    """Render the chapter to PDF with Qt's text engine. Returns False if Qt is unavailable."""
    try:
        from qgis.PyQt.QtCore import QMarginsF, QUrl
        from qgis.PyQt.QtGui import (QFont, QGuiApplication, QImage, QPageLayout, QPageSize, QPdfWriter,
                                     QTextDocument)
    except Exception:
        try:
            from PyQt5.QtCore import QMarginsF, QUrl
            from PyQt5.QtGui import (QFont, QGuiApplication, QImage, QPageLayout, QPageSize, QPdfWriter,
                                     QTextDocument)
        except Exception:
            return _write_pdf_mpl(B, path, accent)      # no Qt (for example ArcGIS Pro): plain matplotlib layout
    if os.environ.get("AREAGIS_PDF_ENGINE") == "mpl":
        return _write_pdf_mpl(B, path, accent)
    app = QGuiApplication.instance()
    if app is None:
        if not os.environ.get("DISPLAY") and os.name != "nt" and not _is_mac():
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        _write_pdf._app = QGuiApplication([])

    def enum(cls, scoped, name):
        """Qt5 (QPageSize.A4) and Qt6/PyQt6 (QPageSize.PageSizeId.A4) spell enums differently."""
        if hasattr(cls, name):
            return getattr(cls, name)
        return getattr(getattr(cls, scoped), name)

    writer = QPdfWriter(path)
    writer.setPageSize(QPageSize(enum(QPageSize, "PageSizeId", "A4")))
    writer.setPageMargins(QMarginsF(18, 16, 18, 16), enum(QPageLayout, "Unit", "Millimeter"))
    writer.setResolution(300)
    writer.setTitle(_creator() + " report")
    doc = QTextDocument()
    f = QFont()
    fams = ["DejaVu Sans", "Segoe UI", "Helvetica Neue", "Arial", "Liberation Sans"]
    if hasattr(f, "setFamilies"):          # Qt >= 5.13: first installed family wins (Windows/macOS/Linux)
        f.setFamilies(fams)
    else:
        f.setFamily("Arial" if os.name == "nt" else fams[0])
    f.setPointSizeF(9.5)
    doc.setDefaultFont(f)
    IMAGE_RES = enum(QTextDocument, "ResourceType", "ImageResource")
    page_w_px = 640  # layout width in document pixels; images are scaled to fit
    doc.setTextWidth(page_w_px)
    doc.setDocumentMargin(0)
    imgs = 0
    h = [f'<html><body style="color:#1c2621">']
    for b in B:
        k = b[0]
        if k == "title":
            h.append(f'<p style="color:{accent};font-size:8pt;font-weight:600">{E(b[2].upper())}</p>'
                     f'<p style="font-family:Georgia,serif;font-size:20pt;font-weight:700;margin-top:2px">{E(b[1])}</p>')
        elif k == "meta":
            h.append(f'<p style="color:#5d6b63;font-size:8.5pt">{E(b[1])}</p>')
        elif k == "grade":
            h.append(f'<p style="color:{LEVEL_COL[b[1]]};font-size:11pt;font-weight:700">Confidence: {b[1]} ({b[2]}/100)</p><hr>')
        elif k in ("h1", "h1x"):
            h.append(f'<h2 style="font-family:Georgia,serif;font-size:14pt;margin-top:14px;color:#1c2621">{E(b[1])}</h2>')
        elif k == "h2":
            h.append(f'<h3 style="font-size:11pt;color:{accent};margin-top:10px">{E(b[1])}</h3>')
        elif k == "p":
            h.append(f"<p>{_bold_html(b[1])}</p>")
        elif k in ("ul", "ol"):
            tag = "ul" if k == "ul" else "ol"
            h.append(f"<{tag}>" + "".join(f"<li>{_bold_html(x)}</li>" for x in b[1]) + f"</{tag}>")
        elif k == "steps":
            h.append("<ol>" + "".join(f"<li><b>{E(t)}.</b> {_bold_html(x)}</li>" for t, x in b[1]) + "</ol>")
        elif k == "kv":
            h.append('<table width="100%" cellspacing="0" cellpadding="5" style="border-collapse:collapse">' + "".join(
                f'<tr><td style="background:#eef1ec;color:#5d6b63">{E(a)}</td><td><b>{_bold_html(v)}</b></td></tr>' for a, v in b[1]) + "</table>")
        elif k == "fig":
            imgs += 1
            name = f"img{imgs}.png"
            doc.addResource(IMAGE_RES, QUrl(name), QImage.fromData(b[1]))
            w, hh = _png_size(b[1])
            ww = page_w_px
            hh2 = int(ww * hh / w)
            cap_h = 430 if (len(b) > 3 and b[3]) else 760
            if hh2 > cap_h:
                ww, hh2 = int(ww * cap_h / hh2), cap_h
            h.append(f'<p align="center"><img src="{name}" width="{ww}" height="{hh2}"></p>'
                     f'<p style="color:#5d6b63;font-style:italic;font-size:8.5pt">{E(b[2])}</p>')
        elif k == "table":
            _, headers, rows, cap, sw = b
            h.append(f'<p style="color:#5d6b63;font-style:italic;font-size:8.5pt;margin-top:8px">{E(cap)}</p>')
            t = ['<table width="100%" cellspacing="0" cellpadding="4" border="0" style="border-collapse:collapse;font-size:8.5pt">']
            t.append("<tr>" + "".join(f'<th align="left" style="background:#e7ece5;border-bottom:1px solid #1c2621">{E(x)}</th>' for x in headers) + "</tr>")
            for i, r in enumerate(rows):
                cells = []
                for j, c in enumerate(r):
                    c = str(c)
                    dot = (f'<span style="color:{sw[i]}">■ </span>' if (sw and j == 0 and i < len(sw) and sw[i]) else "")
                    if c.startswith("http"):
                        c = c.replace("https://", "")
                        cells.append(f'<td style="border-bottom:1px solid #d5dbd2;font-size:6.5pt">{E(c)}</td>')
                    else:
                        cells.append(f'<td style="border-bottom:1px solid #d5dbd2">{dot}{_bold_html(c)}</td>')
                t.append("<tr>" + "".join(cells) + "</tr>")
            t.append("</table>")
            h.append("".join(t))
        elif k == "code":
            h.append(f'<p style="font-family:monospace;font-size:7.5pt;background:#eef1ec">{E(b[1])}</p>')
        elif k == "pagebreak":
            h.append('<p style="page-break-after:always"></p>')
    h.append("</body></html>")
    doc.setHtml("".join(h))
    (doc.print_ if hasattr(doc, "print_") else getattr(doc, "print"))(writer)   # PyQt5: print_, PyQt6: print
    del writer                                                                  # flush the file (Windows locks it)
    return os.path.exists(path) and os.path.getsize(path) > 0


def _write_pdf_mpl(B, path, accent):
    """A4 PDF with matplotlib only, for hosts without Qt. Simpler typography than the Qt layout, same content."""
    if not HAVE_MPL:
        return False
    import textwrap
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.image import imread
    W_, H_ = 8.27, 11.69
    L, R, T, Bm = 0.07, 0.93, 0.95, 0.05
    ink, mute = "#1c2621", "#5d6b63"
    def strip(t):
        return html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"\*\*(.+?)\*\*", r"\1", str(t))))
    st = dict(pdf=PdfPages(path), fig=None, y=T, n=0)

    def new_page():
        if st["fig"] is not None:
            st["fig"].text(0.5, 0.025, str(st["n"]), ha="center", fontsize=7, color=mute)
            st["pdf"].savefig(st["fig"])
        st["fig"] = Figure(figsize=(W_, H_))
        st["y"] = T
        st["n"] += 1

    def need(h):
        if st["fig"] is None or st["y"] - h < Bm:
            new_page()

    def text(t, size=9.0, color=ink, weight="normal", style="normal", width=None, gap=0.004, indent=0.0):
        chars = width or int((R - L - indent) * W_ * 72 / (size * 0.52))
        lines = []
        for para in strip(t).split("\n"):
            lines += textwrap.wrap(para, chars) or [""]
        lh = size * 1.32 / 72 / H_
        for ln in lines:
            need(lh)
            st["fig"].text(L + indent, st["y"], ln, fontsize=size, color=color, weight=weight, style=style, va="top")
            st["y"] -= lh
        st["y"] -= gap

    def table(headers, rows, cap):
        text(cap, 7.5, mute, style="italic")
        ncol = max(1, len(headers))
        lens = [max([len(str(headers[j]))] + [len(strip(r[j])) for r in rows if j < len(r)]) for j in range(ncol)]
        lens = [min(max(x, 4), 40) for x in lens]
        tot = sum(lens)
        xs = [L]
        for x in lens[:-1]:
            xs.append(xs[-1] + (R - L) * x / tot)
        size = 7.0 if ncol <= 5 else 6.0
        lh = size * 1.35 / 72 / H_

        def row(cells, bold=False):
            wrapped = []
            for j, c in enumerate(cells[:ncol]):
                cw = max(4, int((R - L) * lens[j] / tot * W_ * 72 / (size * 0.52)))
                wrapped.append(textwrap.wrap(strip(c).replace("https://", ""), cw)[:4] or [""])
            hgt = lh * max(len(w) for w in wrapped) + 0.003
            need(hgt)
            for j, w in enumerate(wrapped):
                for k, ln in enumerate(w):
                    st["fig"].text(xs[j], st["y"] - k * lh, ln, fontsize=size, va="top", color=ink, weight="bold" if bold else "normal")
            st["y"] -= hgt
            st["fig"].add_artist(__import__("matplotlib").lines.Line2D([L, R], [st["y"] + 0.001] * 2, lw=0.3, color="#c9d0c6"))
        row(list(headers), True)
        for r in rows:
            row([str(c) for c in r])
        st["y"] -= 0.008

    for b in B:
        k = b[0]
        if k == "title":
            text(b[2].upper(), 7.5, accent, "bold")
            text(b[1], 18, ink, "bold", gap=0.006)
        elif k == "meta":
            text(b[1], 8, mute)
        elif k == "grade":
            text(f"Confidence: {b[1]} ({b[2]}/100)", 11, LEVEL_COL.get(b[1], ink), "bold", gap=0.01)
        elif k in ("h1", "h1x"):
            st["y"] -= 0.008
            text(b[1], 13.5, ink, "bold")
        elif k == "h2":
            text(b[1], 10.5, accent, "bold")
        elif k == "p":
            text(b[1])
        elif k in ("ul", "ol"):
            for i, x in enumerate(b[1]):
                text(("• " if k == "ul" else f"{i + 1}. ") + strip(x), indent=0.015, gap=0.001)
            st["y"] -= 0.004
        elif k == "steps":
            for i, (t_, x) in enumerate(b[1]):
                text(f"{i + 1}. {t_}. {strip(x)}", indent=0.015, gap=0.001)
        elif k == "kv":
            table(["Item", "Value"], [[a_, v_] for a_, v_ in b[1]], "")
        elif k == "fig":
            try:
                img = imread(io.BytesIO(b[1]), format="png")
            except Exception as exc:
                skipped(exc)
                continue
            ih, iw = img.shape[:2]
            wfrac = R - L
            hfrac = wfrac * W_ * ih / iw / H_
            cap = 0.42 if (len(b) > 3 and b[3]) else 0.62
            if hfrac > cap:
                wfrac, hfrac = wfrac * cap / hfrac, cap
            need(hfrac + 0.03)
            ax = st["fig"].add_axes([L + (R - L - wfrac) / 2, st["y"] - hfrac, wfrac, hfrac])
            ax.imshow(img); ax.set_axis_off()
            st["y"] -= hfrac + 0.006
            text(b[2], 7.5, mute, style="italic", gap=0.008)
        elif k == "table":
            _, headers, rows, cap, _sw = b
            table(headers, rows, cap)
        elif k == "code":
            text(b[1], 6.5, ink)
        elif k == "pagebreak":
            new_page()
    if st["fig"] is not None:
        st["fig"].text(0.5, 0.025, str(st["n"]), ha="center", fontsize=7, color=mute)
        st["pdf"].savefig(st["fig"])
    st["pdf"].close()
    return os.path.exists(path) and os.path.getsize(path) > 0


def _is_mac():
    import sys
    return sys.platform == "darwin"


def write_deferred_pdf(docs):
    job = docs.pop("pdf_job", None)
    if job and not _write_pdf(*job):
        docs.pop("pdf", None)
    return docs


def _creator():
    try:
        from ..plugin_info import PLUGIN_NAME
        return PLUGIN_NAME
    except Exception:
        return "AreaGIS Kit"
