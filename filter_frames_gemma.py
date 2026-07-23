#!/usr/bin/env python3
"""Filter per-object keyframes down to a clean, non-redundant set (Gemma bbox + MASt3R).

Pipeline per object dir
-----------------------
1. KEEP frames that have a bbox AND are not motion-blurred (Gemma `detections` non-empty
   and the primary detection's `motion_blur` != true; null blur is treated as "keep").
2. AREA-OUTLIER DROP: rank frames by bbox area and automatically discard frames whose
   bbox is significantly smaller than the rest. "Significantly" is a robust lower fence in
   log-area space: drop area < exp(median(logA) - k*1.4826*MAD(logA)). Also drops anything
   below an absolute floor --min-area-frac. Needs >=4 frames for the MAD rule, else floor only.
3. DEDUP BY VIEWPOINT: process survivors largest-bbox-first; keep a frame only if it is NOT a
   near-duplicate viewpoint of any already-kept frame. Duplicate := MASt3R co-visibility
   coverage_B >= --tau-dup with precision >= --min-prec (the validated bg-suppressed crop
   match). Keeping the largest of each viewpoint cluster => clearest representative survives.

Output
------
    <out>/filtered.json              per-frame decisions + kept_frame_ids per dir
    <out>/<dir>/frame_XXXXXX.jpg     the kept (non-duplicate) frames, if --copy-frames

Sharded/resumable like select_new_views.py: --nshards N --shard K, --resume.

Usage
-----
    .venv/bin/python filter_frames_gemma.py --limit 1 --copy-frames      # smoke test
    # full run: 8 shards across 8 GPUs (see the launch note at bottom)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from mast3r.model import AsymmetricMASt3R
import mast3r.utils.path_to_dust3r  # noqa: F401

# reuse the validated co-visibility machinery
from select_new_views import load_mask, warp_overlap, frame_id  # noqa: E402

REPO = Path(__file__).resolve().parent
DEFAULT_BBOX = REPO / "dataset" / "gemma_keyframe_bbox.json"
DEFAULT_KEYFRAMES = REPO / "dataset" / "datatang_session2_videos_fps24_012_keyframes"
DEFAULT_MASKS = REPO / "dataset" / "gemma_keyframe_masks"
DEFAULT_OUT = REPO / "dataset" / "gemma_keyframe_filtered"
DEFAULT_CKPT = REPO / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"


def bbox_area(det: dict) -> float:
    x1, y1, x2, y2 = det["bbox_2d"]
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))


def primary_detection(dets: list) -> dict | None:
    """The largest-area detection = 'the object' for this frame."""
    return max(dets, key=bbox_area) if dets else None


def area_outlier_keep(areas: np.ndarray, area_fracs: np.ndarray, args) -> np.ndarray:
    """Boolean keep-mask: drop significantly-small bbox outliers + absolute floor."""
    keep = area_fracs >= args.min_area_frac
    if keep.sum() >= 4:
        L = np.log(areas[keep])
        m = np.median(L)
        mad = np.median(np.abs(L - m))
        if mad > 0:
            fence = np.exp(m - args.area_mad_k * 1.4826 * mad)
            keep &= areas >= fence
    return keep


def process_dir(name: str, sess: dict, model, args) -> dict:
    kf_dir = args.keyframes_dir / name
    mask_dir = args.mask_dir / name
    frames_meta = sess.get("frames", {})

    # ---- stage 1: bbox present + not blurred ----
    cand = []   # (fid, jpg, area_px, area_frac, blur)
    rows = []
    for fname, fr in sorted(frames_meta.items(), key=lambda kv: frame_id(kv[0])):
        fid = frame_id(fname)
        dets = fr.get("detections", []) or []
        det = primary_detection(dets)
        if det is None:
            rows.append({"frame_id": fid, "decision": "no_bbox"})
            continue
        blur = det.get("motion_blur", None)
        if blur is True:
            rows.append({"frame_id": fid, "decision": "blurred"})
            continue
        W, H = fr.get("width", 1920), fr.get("height", 1080)
        a = bbox_area(det)
        cand.append((fid, kf_dir / fname, a, a / (W * H), bool(blur) if blur is not None else False))

    if not cand:
        return {"keyframe_dir": name, "n_frames": len(frames_meta), "n_candidates": 0,
                "n_kept": 0, "kept_frame_ids": [], "frames": rows}

    # ---- stage 2: area-outlier auto drop ----
    areas = np.array([c[2] for c in cand], float)
    fracs = np.array([c[3] for c in cand], float)
    keep_area = area_outlier_keep(areas, fracs, args)
    survivors = []
    for c, ok in zip(cand, keep_area):
        if not ok:
            rows.append({"frame_id": c[0], "area_frac": round(c[3], 5), "decision": "area_outlier"})
        else:
            survivors.append(c)

    # ---- stage 3: viewpoint dedup, largest bbox first ----
    survivors.sort(key=lambda c: -c[2])
    kept = []   # dicts: {fid, img, mask}
    for fid, jpg, a, af, blur in survivors:
        m = load_mask(mask_dir / f"{jpg.stem}.png")
        if m is None or m.sum() < 50:
            rows.append({"frame_id": fid, "area_frac": round(af, 5), "decision": "no_mask"})
            continue
        img = cv2.cvtColor(cv2.imread(str(jpg)), cv2.COLOR_BGR2RGB)
        dup_of, best_cov, best_prec = None, 0.0, 0.0
        for g in kept:
            s = warp_overlap(g["img"], g["mask"], img, m, model, args)  # warp kept -> candidate
            if s["precision"] >= args.min_prec and s["coverage_B"] >= args.tau_dup:
                dup_of, best_cov, best_prec = g["fid"], s["coverage_B"], s["precision"]
                break
            if s["coverage_B"] > best_cov:
                best_cov, best_prec = s["coverage_B"], s["precision"]
        row = {"frame_id": fid, "area_frac": round(af, 5),
               "max_coverage_vs_kept": round(best_cov, 4), "precision": round(best_prec, 4)}
        if dup_of is not None:
            row["decision"] = "duplicate"
            row["dup_of"] = dup_of
        else:
            row["decision"] = "kept"
            kept.append({"fid": fid, "img": img, "mask": m})
        rows.append(row)

    kept_ids = sorted(k["fid"] for k in kept)
    if args.copy_frames and kept_ids:
        dst = args.out / name
        dst.mkdir(parents=True, exist_ok=True)
        for fid in kept_ids:
            src = kf_dir / f"frame_{fid:06d}.jpg"
            if src.exists():
                shutil.copy2(src, dst / src.name)

    rows.sort(key=lambda r: r["frame_id"])
    return {"keyframe_dir": name, "n_frames": len(frames_meta), "n_candidates": len(cand),
            "n_after_area": len(survivors), "n_kept": len(kept_ids),
            "kept_frame_ids": kept_ids, "frames": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox-json", type=Path, default=DEFAULT_BBOX)
    ap.add_argument("--keyframes-dir", type=Path, default=DEFAULT_KEYFRAMES)
    ap.add_argument("--mask-dir", type=Path, default=DEFAULT_MASKS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--size", type=int, default=512)
    # area outlier
    ap.add_argument("--min-area-frac", type=float, default=0.002, help="absolute bbox-area floor")
    ap.add_argument("--area-mad-k", type=float, default=2.5, help="log-area MAD lower-fence multiplier")
    # dedup
    ap.add_argument("--tau-dup", type=float, default=0.5, help="coverage_B >= this => duplicate viewpoint")
    ap.add_argument("--min-prec", type=float, default=0.5, help="precision gate for a trusted match")
    ap.add_argument("--pad", type=float, default=1.6)
    ap.add_argument("--dilate", type=int, default=7)
    ap.add_argument("--copy-frames", action="store_true", help="copy kept frames into --out/<dir>")
    # ops
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    results = json.loads(args.bbox_json.read_text())["results"]
    items = [(k, v) for k, v in results.items() if v.get("keyframe_dir")]
    items.sort(key=lambda kv: kv[1]["keyframe_dir"])
    if args.nshards > 1:
        items = items[args.shard::args.nshards]
    if args.limit:
        items = items[:args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    rp = args.out / (f"filtered.part{args.shard}of{args.nshards}.json"
                     if args.nshards > 1 else "filtered.json")

    summary, done = [], set()
    if args.resume and rp.exists():
        summary = json.loads(rp.read_text()).get("dirs", [])
        done = {d["keyframe_dir"] for d in summary}
        print(f"resume: {len(done)} dirs done", flush=True)

    def dump():
        rp.write_text(json.dumps({
            "bbox_json": str(args.bbox_json), "mask_dir": str(args.mask_dir),
            "method": "gemma bbox+blur filter -> log-area MAD outlier drop -> "
                      "MASt3R co-visibility viewpoint dedup (largest-bbox-first)",
            "params": {"min_area_frac": args.min_area_frac, "area_mad_k": args.area_mad_k,
                       "tau_dup": args.tau_dup, "min_prec": args.min_prec,
                       "pad": args.pad, "dilate": args.dilate, "size": args.size},
            "num_dirs": len(summary),
            "total_kept": sum(d["n_kept"] for d in summary),
            "dirs": summary,
        }, indent=2))

    todo = [(k, v) for k, v in items if v["keyframe_dir"] not in done]
    print(f"{len(items)} dirs ({len(todo)} to do) | tau_dup={args.tau_dup} "
          f"mad_k={args.area_mad_k} | loading {args.ckpt.name}", flush=True)
    model = AsymmetricMASt3R.from_pretrained(str(args.ckpt)).to(args.device).eval()

    t0 = time.time()
    with torch.no_grad():
        for i, (key, sess) in enumerate(todo, 1):
            name = sess["keyframe_dir"]
            try:
                r = process_dir(name, sess, model, args)
            except Exception as e:
                print(f"[{i}/{len(todo)}] ERROR {name}: {e}", file=sys.stderr, flush=True)
                continue
            summary.append(r)
            dump()
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] {name[:48]:48s} cand={r['n_candidates']:2d} "
                  f"kept={r['n_kept']:2d} {r['kept_frame_ids']}  ({el/i:.1f}s/dir)", flush=True)

    dump()
    print(f"\ndone in {time.time()-t0:.0f}s | {len(summary)} dirs | "
          f"{sum(d['n_kept'] for d in summary)} frames kept\nreport -> {rp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
