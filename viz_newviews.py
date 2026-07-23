#!/usr/bin/env python3
"""Visualize select_new_views.py output: one contact sheet per dir.

For each dir it lays the SELECTED new-view frames left->right as object crops
(mask outlined), labeled with frame id + coverage_B-vs-anchor + decision, so you
can eyeball whether the picks are genuinely distinct viewpoints and whether the
de-duplicated 'same_view' frames were redundant. Optionally appends the skipped
same_view frames as a faded strip (--show-skipped).

Usage
-----
    .venv/bin/python viz_newviews.py                 # all dirs -> <out>/viz/<dir>.png
    .venv/bin/python viz_newviews.py --show-skipped  # also show what was deduped
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

REPO = Path(__file__).resolve().parent
DEFAULT_JSON = REPO / "dataset" / "qwen3vl_keyframe_newviews" / "newviews.json"


def load_mask(path: Path):
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m.max(axis=2)
    return m > 0


def object_crop(img, mask, pad=0.25, out_h=260):
    """Crop a padded box around the mask, draw its contour, return uint8 RGB at out_h."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    px, py = int(bw * pad) + 8, int(bh * pad) + 8
    H, W = img.shape[:2]
    X0, Y0 = max(0, x0 - px), max(0, y0 - py)
    X1, Y1 = min(W, x1 + px), min(H, y1 + py)
    crop = img[Y0:Y1, X0:X1].copy()
    mc = mask[Y0:Y1, X0:X1].astype(np.uint8)
    cnts, _ = cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, cnts, -1, (0, 255, 255), 2)
    scale = out_h / crop.shape[0]
    return cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), out_h))


def make_sheet(d, kf_dir: Path, mask_dir: Path, out_png: Path, show_skipped: bool):
    by_id = {r["frame_id"]: r for r in d["frames"]}
    selected = d["selected_frame_ids"]
    skipped = [r["frame_id"] for r in d["frames"] if r["decision"] == "same_view"] if show_skipped else []
    order = [("sel", f) for f in selected] + ([("skip", f) for f in skipped])
    if not order:
        return False

    tiles = []
    for kind, fid in order:
        jpg = kf_dir / f"frame_{fid:06d}.jpg"
        m = load_mask(mask_dir / f"frame_{fid:06d}.png")
        if not jpg.exists() or m is None:
            continue
        img = cv2.cvtColor(cv2.imread(str(jpg)), cv2.COLOR_BGR2RGB)
        tile = object_crop(img, m)
        if tile is None:
            continue
        r = by_id.get(fid, {})
        cov = r.get("coverage_B")
        dec = r.get("decision", "")
        cap = f"#{fid}  {dec}"
        if cov is not None:
            cap += f"\ncov={cov:.2f} p={r.get('precision',0):.2f}"
        tiles.append((kind, tile, cap))
    if not tiles:
        return False

    n = len(tiles)
    fig, ax = pl.subplots(1, n, figsize=(2.6 * n, 3.4))
    if n == 1:
        ax = [ax]
    for a, (kind, tile, cap) in zip(ax, tiles):
        if kind == "skip":
            tile = (tile.astype(np.float32) * 0.45 + 255 * 0.55).astype(np.uint8)
        a.imshow(tile)
        a.set_title(cap, fontsize=8, color=("black" if kind == "sel" else "gray"))
        for s in a.spines.values():
            s.set_color("crimson" if kind == "sel" else "lightgray")
            s.set_linewidth(2.5 if kind == "sel" else 1)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{d['keyframe_dir']}   |   {d.get('n_area_ok','?')} area-ok  ->  "
                 f"{d['n_selected']} new views (red)"
                 + (f"  |  {len(skipped)} same-view skipped (gray)" if show_skipped else ""),
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=95, bbox_inches="tight")
    pl.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--keyframes-dir", type=Path, default=REPO / "dataset" / "datatang_session2_videos_fps24_012_keyframes")
    ap.add_argument("--mask-dir", type=Path, default=REPO / "dataset" / "qwen3vl_keyframe_masks")
    ap.add_argument("--out", type=Path, default=None, help="viz dir (default: <json parent>/viz)")
    ap.add_argument("--show-skipped", action="store_true", help="also show deduped same-view frames")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    report = json.loads(args.json.read_text())
    dirs = report["dirs"]
    if args.limit:
        dirs = dirs[:args.limit]
    out_dir = args.out or (args.json.parent / "viz")

    made = 0
    for i, d in enumerate(dirs, 1):
        kf = args.keyframes_dir / d["keyframe_dir"]
        mk = args.mask_dir / d["keyframe_dir"]
        if make_sheet(d, kf, mk, out_dir / f"{d['keyframe_dir']}.png", args.show_skipped):
            made += 1
        if i % 25 == 0:
            print(f"  {i}/{len(dirs)} ...")
    print(f"\n{made} contact sheets -> {out_dir}")


if __name__ == "__main__":
    main()
