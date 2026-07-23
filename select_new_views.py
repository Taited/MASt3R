#!/usr/bin/env python3
"""Select frames that show a NEW VIEWPOINT of the masked object, via MASt3R co-visibility.

Motivation
----------
Optical-flow magnitude does NOT tell you about new viewpoints: it measures how far
the object translated in the image (dominated by the hand moving it), not whether a
new surface of the object became visible. What actually answers "is this a new view?"
is CO-VISIBILITY: warp frame A's object-mask pixels into frame B (via MASt3R), rasterize
where they land, and measure how much of B's object mask that covers.

    coverage_B = |warp(A) & mask_B| / |mask_B|
        high -> B's visible surface was already visible in A -> SAME view (redundant)
        low  -> B shows surface A never had                  -> NEW view

    precision  = |warp(A) & mask_B| / |warp(A)|   (TRUST GATE)
        high -> the landed points really sit on B's object -> the low coverage is real
        low  -> points landed off-object -> matching FAILED, coverage is meaningless

Why the crop + background suppression
-------------------------------------
Plain full-frame MASt3R maps an independently-moving object to identity (its rigid-scene
prior treats the object as static background), so it cannot track it at all. Cropping a
window around each frame's own mask AND graying out everything outside the mask on BOTH
crops removes the competing rigid background, so the match becomes object<->object. This
is the only configuration that actually tracks the object (validated empirically).

Selection (anchor-greedy, not consecutive)
------------------------------------------
A slow rotation changes little between consecutive frames but a lot over many frames, so
comparing neighbours misses it. Instead we keep an ANCHOR view and compare each later
frame to it; when coverage drops below --tau-new the frame is a new view and BECOMES the
new anchor. This both scores and de-duplicates: same-view frames are skipped.

Usage
-----
    .venv/bin/python select_new_views.py --limit 1     # smoke test on one dir
    .venv/bin/python select_new_views.py               # full run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r  # noqa: F401  (registers dust3r on sys.path)
from dust3r.inference import inference
from dust3r.utils.image import ImgNorm

REPO = Path(__file__).resolve().parent
DEFAULT_BBOX = REPO / "dataset" / "qwen3vl_keyframe_bbox.json"
DEFAULT_KEYFRAMES = REPO / "dataset" / "datatang_session2_videos_fps24_012_keyframes"
DEFAULT_MASKS = REPO / "dataset" / "qwen3vl_keyframe_masks"
DEFAULT_OUT = REPO / "dataset" / "qwen3vl_keyframe_newviews"
DEFAULT_CKPT = REPO / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

BORDER = 3


def frame_id(name: str) -> int:
    """frame_000123.jpg -> 123"""
    return int(Path(name).stem.split("_")[1])


def load_mask(path: Path) -> np.ndarray | None:
    """Binary object mask (nonzero = object), original resolution."""
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m.max(axis=2)
    return m > 0


def centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    return np.array([xs.mean(), ys.mean()], np.float32)


def view_dict(rgb_uint8: np.ndarray, idx: int) -> dict:
    """Build a dust3r view dict directly from a numpy crop (bypasses load_images,
    which would force a square crop to 4:3 and drop part of the object)."""
    return dict(img=ImgNorm(rgb_uint8)[None], true_shape=np.int32([rgb_uint8.shape[:2]]),
                idx=idx, instance=str(idx))


def crop_window(img: np.ndarray, mask: np.ndarray, win: int):
    """Square win-side crop centered on the mask centroid, clamped inside the image."""
    H, W = img.shape[:2]
    cx, cy = centroid(mask)
    x0 = int(round(np.clip(cx - win / 2, 0, W - win)))
    y0 = int(round(np.clip(cy - win / 2, 0, H - win)))
    return img[y0:y0 + win, x0:x0 + win], np.array([x0, y0], np.float32)


def warp_overlap(imgA, maskA, imgB, maskB, model, args) -> dict:
    """Warp A's masked pixels into B (bg-suppressed crop) and score co-visibility."""
    def side(m):
        ys, xs = np.where(m)
        return max(xs.max() - xs.min(), ys.max() - ys.min())

    win = int(max(args.size, round(args.pad * max(side(maskA), side(maskB)))))
    win = min(win, imgA.shape[0], imgA.shape[1])
    win -= win % 16
    scale = win / args.size

    cA, oA = crop_window(imgA, maskA, win)
    cB, oB = crop_window(imgB, maskB, win)
    cA, cB = cA.copy(), cB.copy()
    # suppress background on BOTH crops -> object<->object match
    mAc = maskA[int(oA[1]):int(oA[1]) + win, int(oA[0]):int(oA[0]) + win]
    mBc = maskB[int(oB[1]):int(oB[1]) + win, int(oB[0]):int(oB[0]) + win]
    cA[~mAc] = 128
    cB[~mBc] = 128
    rA = cv2.resize(cA, (args.size, args.size), interpolation=cv2.INTER_AREA)
    rB = cv2.resize(cB, (args.size, args.size), interpolation=cv2.INTER_AREA)

    out = inference([(view_dict(rA, 0), view_dict(rB, 1))], model, args.device,
                    batch_size=1, verbose=False)
    p1 = {k: v[0] for k, v in out["pred1"].items()}
    p2 = {k: v[0] for k, v in out["pred2"].items()}

    # seed on ALL of A's mask pixels (in net coords)
    ys, xs = np.where(maskA)
    xc, yc = (xs - oA[0]) / scale, (ys - oA[1]) / scale
    ok = (xc >= 0) & (xc < args.size) & (yc >= 0) & (yc < args.size)
    seed = np.unique(np.stack([np.round(xc[ok]).astype(np.int64).clip(0, args.size - 1),
                               np.round(yc[ok]).astype(np.int64).clip(0, args.size - 1)], 1), axis=0)
    if len(seed) == 0:
        return {"iou": 0.0, "coverage_B": 0.0, "precision": 0.0, "match_ratio": 0.0, "n": 0}

    p, q = fast_reciprocal_NNs(p1["desc"].detach(), p2["desc"].detach(),
                               subsample_or_initxy1=(seed[:, 0], seed[:, 1]),
                               device=args.device, dist="dot", block_size=2**13)
    b = BORDER
    keep = ((q[:, 0] >= b) & (q[:, 0] < args.size - b) & (q[:, 1] >= b) & (q[:, 1] < args.size - b) &
            (p[:, 0] >= b) & (p[:, 0] < args.size - b) & (p[:, 1] >= b) & (p[:, 1] < args.size - b))
    q = q[keep]
    match_ratio = len(q) / len(seed)
    q_orig = q * scale + oB

    # rasterize landed points -> warped mask, dilate to fill sub-sampling gaps
    H, W = maskB.shape
    Wm = np.zeros((H, W), np.uint8)
    if len(q_orig):
        qi = np.round(q_orig).astype(np.int64)
        qi[:, 0] = qi[:, 0].clip(0, W - 1)
        qi[:, 1] = qi[:, 1].clip(0, H - 1)
        Wm[qi[:, 1], qi[:, 0]] = 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate, args.dilate))
        Wm = cv2.dilate(Wm, k)
    Wm = Wm > 0

    inter = int((Wm & maskB).sum())
    union = int((Wm | maskB).sum())
    return {
        "iou": inter / union if union else 0.0,
        "coverage_B": inter / int(maskB.sum()) if maskB.sum() else 0.0,
        "precision": inter / int(Wm.sum()) if Wm.sum() else 0.0,
        "match_ratio": float(match_ratio),
        "n": int(len(q_orig)),
    }


def process_dir(kf_dir: Path, mask_dir: Path, model, args) -> dict:
    frames = sorted(kf_dir.glob("frame_*.jpg"), key=lambda p: frame_id(p.name))

    # ---- stage 1: load masks + area filter ----
    entries = []          # (fid, jpg_path, mask, area_frac)
    for f in frames:
        m = load_mask(mask_dir / f"{f.stem}.png")
        if m is None:
            continue
        entries.append([frame_id(f.name), f, m, float(m.sum()) / m.size])
    if not entries:
        return {"keyframe_dir": kf_dir.name, "n_frames": 0, "n_area_ok": 0,
                "median_area_frac": 0.0, "n_selected": 0,
                "selected_frame_ids": [], "frames": []}

    med_area = float(np.median([e[3] for e in entries]))
    area_ok = [e[3] >= args.min_area_frac and e[3] >= args.rel_area * med_area for e in entries]

    rows = []
    good = [(e, ok) for e, ok in zip(entries, area_ok)]

    # ---- stage 2: anchor-greedy new-view selection over area-passing frames ----
    anchor = None         # (fid, img, mask)
    selected = []
    for (fid, jpg, mask, af), ok in good:
        if not ok:
            rows.append({"frame_id": fid, "area_frac": af, "decision": "dropped_small"})
            continue
        img = cv2.cvtColor(cv2.imread(str(jpg)), cv2.COLOR_BGR2RGB)
        if anchor is None:
            anchor = (fid, img, mask)
            selected.append(fid)
            rows.append({"frame_id": fid, "area_frac": af, "anchor_id": fid,
                         "decision": "anchor"})
            continue

        s = warp_overlap(anchor[1], anchor[2], img, mask, model, args)
        row = {"frame_id": fid, "area_frac": af, "anchor_id": anchor[0],
               "coverage_B": round(s["coverage_B"], 4), "precision": round(s["precision"], 4),
               "iou": round(s["iou"], 4), "match_ratio": round(s["match_ratio"], 4)}
        if s["precision"] < args.min_prec:
            # cannot trust the score -> not a viewpoint judgement
            row["decision"] = "low_conf"
            if args.keep_low_conf:
                selected.append(fid); anchor = (fid, img, mask); row["decision"] = "low_conf_kept"
        elif s["coverage_B"] < args.tau_new:
            row["decision"] = "new_view"
            selected.append(fid)
            anchor = (fid, img, mask)      # the new view becomes the reference
        else:
            row["decision"] = "same_view"
        rows.append(row)

    return {"keyframe_dir": kf_dir.name, "n_frames": len(entries),
            "n_area_ok": int(sum(area_ok)), "median_area_frac": round(med_area, 5),
            "n_selected": len(selected), "selected_frame_ids": selected, "frames": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox-json", type=Path, default=DEFAULT_BBOX)
    ap.add_argument("--keyframes-dir", type=Path, default=DEFAULT_KEYFRAMES)
    ap.add_argument("--mask-dir", type=Path, default=DEFAULT_MASKS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--size", type=int, default=512)
    # area filter
    ap.add_argument("--min-area-frac", type=float, default=0.004,
                    help="drop frames whose mask covers < this fraction of the image")
    ap.add_argument("--rel-area", type=float, default=0.3,
                    help="also drop frames with mask < this * median mask area in the dir")
    # new-view decision
    ap.add_argument("--tau-new", type=float, default=0.4,
                    help="coverage_B below this vs the anchor => new viewpoint")
    ap.add_argument("--min-prec", type=float, default=0.5,
                    help="precision gate; below this the match is untrusted (low_conf)")
    ap.add_argument("--keep-low-conf", action="store_true",
                    help="also select low_conf frames (conservative; may keep failures)")
    # matching / crop
    ap.add_argument("--pad", type=float, default=1.6, help="crop side = pad * longest bbox side")
    ap.add_argument("--dilate", type=int, default=7, help="dilate warped points into a mask (px)")
    ap.add_argument("--limit", type=int, default=0, help="only process N dirs (debug)")
    ap.add_argument("--resume", action="store_true",
                    help="skip dirs already present in an existing newviews.json")
    ap.add_argument("--nshards", type=int, default=1, help="split dirs across N processes")
    ap.add_argument("--shard", type=int, default=0, help="this process's shard index [0,nshards)")
    args = ap.parse_args()

    results = json.loads(args.bbox_json.read_text())["results"]
    dirs = [v["keyframe_dir"] for v in results.values() if v.get("keyframe_dir")]
    dirs = sorted(set(dirs))
    if args.nshards > 1:
        dirs = dirs[args.shard::args.nshards]
    if args.limit:
        dirs = dirs[:args.limit]
    if not dirs:
        ap.error("no keyframe dirs to process")

    args.out.mkdir(parents=True, exist_ok=True)
    rp = args.out / (f"newviews.part{args.shard}of{args.nshards}.json"
                     if args.nshards > 1 else "newviews.json")

    # resume: carry over dirs already computed
    summary = []
    done = set()
    if args.resume and rp.exists():
        prev = json.loads(rp.read_text()).get("dirs", [])
        summary = prev
        done = {d["keyframe_dir"] for d in prev}
        print(f"resume: {len(done)} dirs already done, skipping them")

    def dump():
        rp.write_text(json.dumps({
            "keyframes_dir": str(args.keyframes_dir),
            "mask_dir": str(args.mask_dir),
            "method": "MASt3R co-visibility (warp A mask into B on bg-suppressed crops); "
                      "coverage_B < tau_new & precision >= min_prec => new viewpoint; "
                      "anchor-greedy selection",
            "params": {"tau_new": args.tau_new, "min_prec": args.min_prec,
                       "min_area_frac": args.min_area_frac, "rel_area": args.rel_area,
                       "pad": args.pad, "dilate": args.dilate, "size": args.size},
            "num_dirs": len(summary),
            "total_selected": sum(s["n_selected"] for s in summary),
            "dirs": summary,
        }, indent=2))

    todo = [n for n in dirs if n not in done]
    print(f"{len(dirs)} dirs ({len(todo)} to do) | tau_new={args.tau_new} "
          f"min_prec={args.min_prec} min_area={args.min_area_frac} | loading {args.ckpt.name}")
    model = AsymmetricMASt3R.from_pretrained(str(args.ckpt)).to(args.device).eval()

    t0 = time.time()
    with torch.no_grad():
        for i, name in enumerate(todo, 1):
            kf_dir = args.keyframes_dir / name
            mask_dir = args.mask_dir / name
            if not kf_dir.is_dir() or not mask_dir.is_dir():
                continue
            try:
                r = process_dir(kf_dir, mask_dir, model, args)
            except Exception as e:
                print(f"[{i}/{len(todo)}] ERROR {name}: {e}", file=sys.stderr)
                continue
            summary.append(r)
            dump()                          # checkpoint after every dir
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] {name[:50]:50s} "
                  f"area_ok={r.get('n_area_ok', 0):3d} sel={r['n_selected']:2d} "
                  f"{r['selected_frame_ids']}  ({el/i:.1f}s/dir)", flush=True)

    dump()
    print(f"\ndone in {time.time()-t0:.0f}s | {len(summary)} dirs total | "
          f"{sum(s['n_selected'] for s in summary)} new-view frames\nreport -> {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
