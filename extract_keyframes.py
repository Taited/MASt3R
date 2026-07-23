#!/usr/bin/env python3
"""Keyframe extraction by inter-frame difference.

For each video: decode every frame at native fps, score each frame by how much
it differs from its predecessor, and keep only the most-different `--keep-ratio`
fraction (default 10%, i.e. drop the 90% lowest-difference frames).

Difference metric (--metric mse, default):
    frame -> grayscale -> resize to 128x128 (INTER_AREA) -> MSE vs previous frame

The downscale is the important part: at 1920x1080 a raw MSE is dominated by
sensor noise and H.264 block artifacts rather than real content change, and
INTER_AREA area-averaging suppresses both while keeping structural motion. It
also makes scores comparable across resolutions and is ~200x cheaper.

Selection is rank-based (top-k by difference) rather than a raw `diff > t`
comparison. The two are equivalent for a smooth distribution, but on a nearly
static video many diffs are identical, and a threshold comparison would then
keep either none or all of them. Rank-based selection always yields exactly k.
The equivalent threshold (the k-th largest diff) is recorded in the manifest.

Frame ids are the 0-based decode index within the mp4, zero-padded to 6 digits,
so `frame_000123.jpg` is exactly the 124th decoded frame of the source mp4.
NOTE: the `frames_64-242` part of the source filenames does NOT agree with the
actual mp4 frame counts, so it is not used for numbering.

Output layout:
    <outdir>/
        <video_stem>/frame_%06d.jpg     # <video_stem>.mp4 is the source
        manifest.json                   # per-video fps/threshold/kept ids

Usage:
    .venv/bin/python extract_keyframes.py
    .venv/bin/python extract_keyframes.py --keep-ratio 0.1 --jobs 16
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
DEFAULT_SRC = REPO / "dataset" / "datatang_session2_videos_fps24_012"
DEFAULT_OUT = REPO / "dataset" / "datatang_session2_videos_fps24_012_keyframes"

PROXY_SIZE = 128  # side length of the grayscale proxy used for scoring


def _proxy(frame: np.ndarray) -> np.ndarray:
    """Downscaled grayscale proxy used for difference scoring."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (PROXY_SIZE, PROXY_SIZE), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def _score(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    d = b - a
    if metric == "mse":
        return float(np.mean(d * d))
    if metric == "mad":
        return float(np.mean(np.abs(d)))
    raise ValueError(f"unknown metric: {metric}")


def score_video(path: Path, metric: str) -> tuple[list[float], dict]:
    """Pass 1: decode every frame, score each against its predecessor.

    Returns (diffs, meta) where diffs[i] is the difference between frame i and
    frame i-1. diffs[0] is inf (no predecessor; frame 0 is the anchor).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    meta = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    diffs: list[float] = []
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cur = _proxy(frame)
        diffs.append(math.inf if prev is None else _score(prev, cur, metric))
        prev = cur
    cap.release()
    meta["total_frames"] = len(diffs)
    return diffs, meta


def select(diffs: list[float], keep_ratio: float, keep_first: bool) -> tuple[list[int], float]:
    """Pick the top-k frame ids by difference. Returns (sorted ids, threshold)."""
    n = len(diffs)
    if n == 0:
        return [], float("nan")
    if n == 1:
        return [0], float("nan")

    # frame 0 has no predecessor, so it is not part of the difference population
    scored = np.asarray(diffs[1:], dtype=np.float64)
    ids = np.arange(1, n)

    k = int(math.ceil(keep_ratio * scored.size))
    k = max(1, min(k, scored.size))

    order = np.argsort(-scored, kind="stable")  # descending
    top = order[:k]
    threshold = float(scored[order[k - 1]])  # k-th largest == effective cutoff

    kept = set(ids[top].tolist())
    if keep_first:
        kept.add(0)
    return sorted(kept), threshold


def write_frames(path: Path, outdir: Path, kept: list[int], quality: int) -> int:
    """Pass 2: re-decode and write only the selected frames as JPEG."""
    outdir.mkdir(parents=True, exist_ok=True)
    want = set(kept)
    cap = cv2.VideoCapture(str(path))
    idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            dst = outdir / f"frame_{idx:06d}.jpg"
            if not cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                raise RuntimeError(f"failed writing {dst}")
            written += 1
        idx += 1
    cap.release()
    return written


def process(path_str: str, out_root_str: str, keep_ratio: float, metric: str,
            keep_first: bool, quality: int) -> dict:
    path = Path(path_str)
    stem = path.stem
    diffs, meta = score_video(path, metric)
    kept, threshold = select(diffs, keep_ratio, keep_first)
    written = write_frames(path, Path(out_root_str) / stem, kept, quality)
    finite = [d for d in diffs[1:] if math.isfinite(d)]
    return {
        "video": path.name,
        "dir": stem,
        **meta,
        "metric": metric,
        "threshold": threshold,
        "diff_min": min(finite) if finite else None,
        "diff_median": float(np.median(finite)) if finite else None,
        "diff_max": max(finite) if finite else None,
        "kept_count": written,
        "keep_ratio_actual": written / meta["total_frames"] if meta["total_frames"] else 0.0,
        "kept_frame_ids": kept,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--keep-ratio", type=float, default=0.10,
                    help="fraction of frames to keep (default 0.10 = drop 90%%)")
    ap.add_argument("--metric", choices=["mse", "mad"], default="mse")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--no-keep-first", dest="keep_first", action="store_false",
                    help="do not force-keep frame 0 of each video")
    args = ap.parse_args()

    if not 0 < args.keep_ratio <= 1:
        ap.error("--keep-ratio must be in (0, 1]")

    videos = sorted(args.src.glob("*.mp4"))
    if not videos:
        ap.error(f"no .mp4 found under {args.src}")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(videos)} videos | metric={args.metric} keep_ratio={args.keep_ratio} "
          f"jobs={args.jobs}\nout: {args.out}\n")

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {
            ex.submit(process, str(v), str(args.out), args.keep_ratio, args.metric,
                      args.keep_first, args.jpeg_quality): v
            for v in videos
        }
        for i, fut in enumerate(as_completed(futs), 1):
            v = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # keep going; report at the end
                print(f"[{i}/{len(videos)}] ERROR {v.name}: {e}", file=sys.stderr)
                continue
            results.append(r)
            print(f"[{i}/{len(videos)}] {r['dir'][:58]:58s} "
                  f"{r['kept_count']:3d}/{r['total_frames']:4d} "
                  f"({r['keep_ratio_actual']*100:4.1f}%)  t={r['threshold']:.3f}")

    results.sort(key=lambda r: r["video"])
    total = sum(r["total_frames"] for r in results)
    kept = sum(r["kept_count"] for r in results)
    manifest = {
        "source_dir": str(args.src),
        "frame_id_convention": "0-based decode index within the source mp4",
        "metric": args.metric,
        "proxy": f"grayscale {PROXY_SIZE}x{PROXY_SIZE} INTER_AREA",
        "keep_ratio_requested": args.keep_ratio,
        "threshold_scope": "per-video (90th percentile of that video's diffs)",
        "keep_first_frame": args.keep_first,
        "jpeg_quality": args.jpeg_quality,
        "num_videos": len(results),
        "total_frames": total,
        "total_kept": kept,
        "keep_ratio_actual": kept / total if total else 0.0,
        "videos": results,
    }
    mpath = args.out / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))

    print(f"\ndone: {len(results)}/{len(videos)} videos | kept {kept}/{total} frames "
          f"({kept/total*100:.2f}%)\nmanifest -> {mpath}")
    return 0 if len(results) == len(videos) else 1


if __name__ == "__main__":
    sys.exit(main())
