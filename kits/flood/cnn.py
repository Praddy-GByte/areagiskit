# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Small U-Net flood classifier, run with numpy only (no PyTorch needed in QGIS).

Trained on 52 floods mapped by Copernicus EMS analysts (validation/floodbench, train_cnn.py).
Leave-event-out cross-validation: mean CSI 0.46 vs 0.39 for GFM alone and 0.44 for GFM + change detection.
Inputs (19 channels) are built by `features`; the weights are in cnn_weights.npz next to this file.
"""

import os

import numpy as np

_W = None


def _weights():
    global _W
    if _W is None:
        z = np.load(os.path.join(os.path.dirname(__file__), "cnn_weights.npz"))
        _W = {k: z[k].astype(np.float32) for k in z.files}
    return _W


def features(vv, vh, vv_ly, vh_ly, vv_pre, vh_pre, hand, slope, ens, dlr, tuw, lst, like, excl, occ, wc, valid):
    """Same channel order and scaling as validation/floodbench/prep.py."""
    def z(a):
        return np.nan_to_num(a, nan=0.0).astype(np.float32)
    ly = vv_ly if (vv_ly is not None and np.isfinite(vv_ly).any()) else vv_pre
    lyh = vh_ly if (vh_ly is not None and np.isfinite(vh_ly).any()) else vh_pre
    pr = vv_pre if (vv_pre is not None and np.isfinite(vv_pre).any()) else ly
    prh = vh_pre if (vh_pre is not None and np.isfinite(vh_pre).any()) else lyh
    L = like.astype(np.float32)
    L[L > 100] = 0
    n = (dlr == 1).astype(np.float32) + (tuw == 1) + (lst == 1)
    F = [(z(vv) + 15) / 7, (z(vh) + 22) / 7, np.clip(z(vv - ly), -15, 15) / 5, np.clip(z(vh - lyh), -15, 15) / 5,
         np.clip(z(vv - pr), -15, 15) / 5, np.clip(z(vh - prh), -15, 15) / 5,
         np.clip(np.nan_to_num(hand, nan=30), 0, 30) / 10, np.clip(z(slope), 0, 30) / 10,
         (ens == 1).astype(np.float32), n / 3, L / 100, (excl == 1).astype(np.float32), z(occ) / 100,
         (wc == 2).astype(np.float32), (wc == 5).astype(np.float32), (wc == 7).astype(np.float32), (wc == 11).astype(np.float32),
         (wc == 1).astype(np.float32), valid.astype(np.float32)]
    X = np.stack(F).astype(np.float32)
    X[:, ~valid] = 0
    return X


def _conv(x, w, b, relu=True):
    """3x3 (or 1x1) convolution, padding 'same'. x: (C, H, W); w: (O, C, k, k)."""
    O, C, k, _ = w.shape
    _, H, W = x.shape
    p = k // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p))) if p else x
    out = np.zeros((O, H * W), np.float32)
    for dy in range(k):
        for dx in range(k):
            patch = xp[:, dy:dy + H, dx:dx + W].reshape(C, -1)
            out += w[:, :, dy, dx] @ patch
    out += b[:, None]
    out = out.reshape(O, H, W)
    return np.maximum(out, 0) if relu else out


def _seq(x, W, name, n=2):
    for i in (0, 2)[:n]:
        x = _conv(x, W[f"{name}.{i}.weight"], W[f"{name}.{i}.bias"])
    return x


def _pool(x):
    C, H, W = x.shape
    return x[:, :H // 2 * 2, :W // 2 * 2].reshape(C, H // 2, 2, W // 2, 2).max((2, 4))


def _up(x):
    return x.repeat(2, 1).repeat(2, 2)


def forward(x):
    W = _weights()
    a = _seq(x, W, "e1")
    b = _seq(_pool(a), W, "e2")
    c = _seq(_pool(b), W, "b")
    d = _seq(np.concatenate([_up(c), b], 0), W, "d2")
    e = _seq(np.concatenate([_up(d), a], 0), W, "d1")
    lo = _conv(e, W["out.weight"], W["out.bias"], relu=False)[0]
    return 1 / (1 + np.exp(-np.clip(lo, -30, 30)))


def predict(X, tile=384, pad=16, progress=None):
    C, H, W = X.shape
    out = np.zeros((H, W), np.float32)
    n = ((H + tile - 1) // tile) * ((W + tile - 1) // tile)
    i = 0
    for r in range(0, H, tile):
        for c in range(0, W, tile):
            r0, c0 = max(r - pad, 0), max(c - pad, 0)
            r1, c1 = min(r + tile + pad, H), min(c + tile + pad, W)
            xx = X[:, r0:r1, c0:c1]
            h, w = xx.shape[1:]
            xx = np.pad(xx, ((0, 0), (0, (-h) % 4), (0, (-w) % 4)))
            p = forward(xx)[:h, :w]
            out[r:min(r + tile, H), c:min(c + tile, W)] = p[r - r0:r - r0 + min(tile, H - r), c - c0:c - c0 + min(tile, W - c)]
            i += 1
            if progress and i % 10 == 0:
                progress(i, n)
    return out
