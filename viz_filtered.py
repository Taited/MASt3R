#!/usr/bin/env python3
"""Visualize filter_frames_gemma.py output: per-dir sheet of KEPT (non-duplicate) frames.

Top row = kept frames (green border, labeled frame id). Optionally a second faded row of
frames dropped as duplicates, each labeled with the kept frame it duplicates (--show-dropped).

Usage
-----
    .venv/bin/python viz_filtered.py                 # all dirs -> <out>/viz/<dir>.png
    .venv/bin/python viz_filtered.py --show-dropped  # also show duplicates + why
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

from viz_newviews import load_mask, object_crop   # reuse helpers

REPO = Path(__file__).resolve().parent
DEFAULT_JSON = REPO / "dataset" / "gemma_keyframe_filtered" / "filtered.json"


def make_sheet(d, kf_dir: Path, mask_dir: Path, out_png: Path, show_dropped: bool):
    kept = d["kept_frame_ids"]
    dups = [f for f in d["frames"] if f["decision"] == "duplicate"] if show_dropped else []
    by_id = {f["frame_id"]: f for f in d["frames"]}
    order = [("keep", f) for f in kept] + [("dup", f["frame_id"]) for f in dups]
    if not order:
        return False

    tiles = []
    for kind, fid in order:
        jpg = kf_dir / f"frame_{fid:06d}.jpg"
        m = load_mask(mask_dir / f"frame_{fid:06d}.png")
        if not jpg.exists() or m is None or m.sum() < 50:
            continue
        img = cv2.cvtColor(cv2.imread(str(jpg)), cv2.COLOR_BGR2RGB)
        tile = object_crop(img, m)
        if tile is None:
            continue
        r = by_id.get(fid, {})
        if kind == "keep":
            cap = f"#{fid}  KEPT"
            cov = r.get("max_coverage_vs_kept")
            if cov is not None:
                cap += f"\ncov_vs_kept={cov:.2f}"
        else:
            cap = f"#{fid}  dup of {r.get('dup_of')}\ncov={r.get('max_coverage_vs_kept'):.2f}"
        tiles.append((kind, tile, cap))
    if not tiles:
        return False

    n = len(tiles)
    fig, ax = pl.subplots(1, n, figsize=(2.6 * n, 3.4))
    if n == 1:
        ax = [ax]
    for a, (kind, tile, cap) in zip(ax, tiles):
        if kind == "dup":
            tile = (tile.astype(np.float32) * 0.45 + 255 * 0.55).astype(np.uint8)
        a.imshow(tile)
        a.set_title(cap, fontsize=8, color=("black" if kind == "keep" else "gray"))
        for s in a.spines.values():
            s.set_color("green" if kind == "keep" else "lightgray")
            s.set_linewidth(2.5 if kind == "keep" else 1)
        a.set_xticks([]); a.set_yticks([])
    nb = sum(1 for f in d["frames"] if f["decision"] == "blurred")
    na = sum(1 for f in d["frames"] if f["decision"] == "area_outlier")
    nd = sum(1 for f in d["frames"] if f["decision"] == "duplicate")
    fig.suptitle(f"{d['keyframe_dir']}   |   {d['n_candidates']} cand -> {d['n_kept']} kept  "
                 f"(dropped: {nb} blur, {na} small, {nd} dup)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=95, bbox_inches="tight")
    pl.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--keyframes-dir", type=Path, default=REPO / "dataset" / "datatang_session2_videos_fps24_012_keyframes")
    ap.add_argument("--mask-dir", type=Path, default=REPO / "dataset" / "gemma_keyframe_masks")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--show-dropped", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dirs = json.loads(args.json.read_text())["dirs"]
    if args.limit:
        dirs = dirs[:args.limit]
    out_dir = args.out or (args.json.parent / "viz")
    made = 0
    for i, d in enumerate(dirs, 1):
        kf = args.keyframes_dir / d["keyframe_dir"]
        mk = args.mask_dir / d["keyframe_dir"]
        if make_sheet(d, kf, mk, out_dir / f"{d['keyframe_dir']}.png", args.show_dropped):
            made += 1
        if i % 25 == 0:
            print(f"  {i}/{len(dirs)} ...")
    print(f"\n{made} contact sheets -> {out_dir}")


if __name__ == "__main__":
    main()
