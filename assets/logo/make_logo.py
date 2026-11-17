# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""AreaGIS Kit logo: one area, six studies."""
import math
COL = ["#2D7FD3", "#2E9E6A", "#B5712E", "#9BB52A", "#E04B3A", "#8E44AD"]   # flood, lulc, landslide, crop, urban, health
NAVY, NAVY2 = "#0B2A42", "#123B5C"
C = (256, 262)
PTS = [(256, 96), (402, 170), (416, 334), (288, 424), (122, 376), (104, 198)]


def mark(bg=True, small=False):
    w = []
    if bg:
        w.append(f'<rect x="0" y="0" width="512" height="512" rx="112" fill="{NAVY}"/>')
        w.append(f'<rect x="20" y="20" width="472" height="472" rx="96" fill="none" stroke="{NAVY2}" stroke-width="4"/>')
    poly = " ".join(f"{x},{y}" for x, y in PTS)
    # dashed outline around the area, slightly larger: the "selected area"
    big = []
    for x, y in PTS:
        dx, dy = x - C[0], y - C[1]
        big.append(f"{C[0] + dx * 1.12:.1f},{C[1] + dy * 1.12:.1f}")
    if not small:
        w.append(f'<polygon points="{" ".join(big)}" fill="none" stroke="#FFFFFF" stroke-opacity="0.55" '
                 f'stroke-width="7" stroke-dasharray="18 14" stroke-linejoin="round"/>')
    if small:   # bigger area, bolder lines for 16-32 px toolbar icons
        PT = [(C[0] + (x - C[0]) * 1.18, C[1] + (y - C[1]) * 1.18) for x, y in PTS]
    else:
        PT = PTS
    sw = 2 if small else 1
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in PT)
    for i in range(6):
        a, b = PT[i], PT[(i + 1) % 6]
        w.append(f'<polygon points="{C[0]},{C[1]} {a[0]},{a[1]} {b[0]},{b[1]}" fill="{COL[i]}"/>')
    for x, y in PT:   # thin white seams between the six studies
        w.append(f'<line x1="{C[0]}" y1="{C[1]}" x2="{x}" y2="{y}" stroke="#FFFFFF" stroke-width="{6 * sw}" stroke-linecap="round"/>')
    w.append(f'<polygon points="{poly}" fill="none" stroke="#FFFFFF" stroke-width="{10 * sw}" stroke-linejoin="round"/>')
    w.append(f'<circle cx="{C[0]}" cy="{C[1]}" r="{36 if small else 30}" fill="#FFFFFF"/>')
    if not small:
        w.append(f'<circle cx="{C[0]}" cy="{C[1]}" r="13" fill="{NAVY}"/>')
    return "\n  ".join(w)


def svg(inner, w, h, vb=None):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="{vb or f"0 0 {w} {h}"}">\n  {inner}\n</svg>\n')


icon = svg(mark(), 512, 512)
open("areagiskit_icon.svg", "w").write(icon)
open("areagiskit_icon_small.svg", "w").write(svg(mark(small=True), 512, 512))


def lockup(dark):
    fg = "#FFFFFF" if dark else NAVY
    sub = "#9FB6CA" if dark else "#4A6378"
    bg = f'<rect width="1400" height="420" fill="{NAVY}"/>' if dark else '<rect width="1400" height="420" fill="#FFFFFF"/>'
    g = f'<g transform="translate(60,50) scale(0.625)">{mark()}</g>'
    t = (f'<text x="440" y="218" font-family="Inter" font-weight="800" font-size="150" fill="{fg}" letter-spacing="-4">AreaGIS'
         f'<tspan font-weight="300" dx="22">Kit</tspan></text>'
         f'<text x="446" y="300" font-family="Inter" font-weight="500" font-size="42" fill="{sub}" letter-spacing="1">'
         f'One area. Six studies. 236 analyses.</text>')
    return svg(bg + g + t, 1400, 420)


open("areagiskit_logo_light.svg", "w").write(lockup(False))
open("areagiskit_logo_dark.svg", "w").write(lockup(True))

# GitHub social preview 1280x640
names = ["Flood", "LULC", "Landslide", "Crop", "Urban", "Health"]
chips = ""
x = 120
for n, c in zip(names, COL):
    wdt = 38 + 22 * len(n)
    chips += (f'<rect x="{x}" y="452" width="{wdt}" height="54" rx="27" fill="{c}"/>'
              f'<text x="{x + wdt / 2}" y="488" text-anchor="middle" font-family="Inter" font-weight="700" font-size="26" fill="#FFFFFF">{n}</text>')
    x += wdt + 16
soc = (f'<rect width="1280" height="640" fill="{NAVY}"/>'
       f'<g transform="translate(985,110) scale(0.52)">{mark(bg=False)}</g>'
       f'<text x="120" y="250" font-family="Inter" font-weight="800" font-size="118" fill="#FFFFFF" letter-spacing="-3">AreaGIS'
       f'<tspan font-weight="300" dx="16">Kit</tspan></text>'
       f'<text x="124" y="320" font-family="Inter" font-weight="500" font-size="36" fill="#9FB6CA">One area. Six studies. 236 analyses.</text>'
       f'<text x="124" y="372" font-family="Inter" font-weight="500" font-size="28" fill="#6F8BA3">Open source  ·  QGIS + ArcGIS Pro  ·  Word, PDF, HTML reports</text>'
       + chips)
open("areagiskit_social.svg", "w").write(svg(soc, 1280, 640))
