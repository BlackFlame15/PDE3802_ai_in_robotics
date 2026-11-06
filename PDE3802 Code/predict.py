#!/usr/bin/env python3
"""Classify office items from image file(s), a folder, a video file or a live camera.

  python predict.py --weights best.pt --source photo.jpg
  python predict.py --weights best.pt --source my_photos/ --csv results.csv
  python predict.py --weights best.pt --source 0              # webcam index 0
  python predict.py --weights best.pt --source 0 --imgsz 320 --center-crop 0.7

One item at a time: the whole frame (or the centre crop, if --center-crop is
given) is treated as showing a single item. Output: predicted class and
confidence (temperature-calibrated if calibration.json sits next to the weights).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

from office_cls import OfficeClassifier, list_images, read_image


def center_crop(frame, frac):
    h, w = frame.shape[:2]
    s = int(min(h, w) * frac)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return frame[y0:y0 + s, x0:x0 + s], (x0, y0, x0 + s, y0 + s)


def run_images(clf, paths, args):
    rows = []
    for p in paths:
        img = read_image(p)
        if img is None:
            print(f"skip (unreadable): {p}", file=sys.stderr)
            continue
        if args.center_crop:
            img, _ = center_crop(img, args.center_crop)
        res = clf.classify(img, args.topk, args.min_conf)
        rows.append((str(p), res["class"], res["confidence"]))
        if args.json:
            print(json.dumps({"source": str(p), **res}))
        else:
            print(f"{p}\t{res['class']}\t{res['confidence']:.3f}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source", "class", "confidence"])
            w.writerows(rows)


def run_stream(clf, source, args):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera/video: {source}")
    smoothed, n, t_prev = None, 0, time.perf_counter()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            view, box = (center_crop(frame, args.center_crop) if args.center_crop else (frame, None))
            p = clf.probs(view)
            # Exponential moving average steadies the label between frames.
            smoothed = p if smoothed is None or args.smooth <= 0 else args.smooth * smoothed + (1 - args.smooth) * p
            res = clf.describe(smoothed, args.topk, args.min_conf)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if n else 1 / max(now - t_prev, 1e-6)
            t_prev, n = now, n + 1
            if args.json:
                print(json.dumps({"frame": n, **res}), flush=True)
            elif n % args.print_every == 0:
                print(f"{res['class']}\t{res['confidence']:.3f}\t{fps:.1f} FPS", flush=True)
            if not args.no_display:
                if box:
                    cv2.rectangle(frame, box[:2], box[2:], (0, 255, 0), 2)
                cv2.putText(frame, f"{res['class']} {res['confidence']:.2f}  {fps:.1f} FPS", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("office classifier (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help=".pt / .onnx / NCNN model directory")
    ap.add_argument("--source", required=True, help="image, folder, video file, or camera index (e.g. 0)")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--min-conf", type=float, default=0.0, help="below this, report 'unsure'")
    ap.add_argument("--temperature", type=float, default=None, help="override calibration.json")
    ap.add_argument("--center-crop", type=float, default=0.0, help="use the central square (fraction of short side)")
    ap.add_argument("--smooth", type=float, default=0.6, help="EMA factor for live input (0 = off)")
    ap.add_argument("--csv", help="write results for file/folder input to this CSV")
    ap.add_argument("--json", action="store_true", help="print one JSON object per result")
    ap.add_argument("--no-display", action="store_true", help="no window (headless / SSH)")
    ap.add_argument("--print-every", type=int, default=10, help="live mode: print every N frames")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    clf = OfficeClassifier(args.weights, args.imgsz, args.temperature, args.device)
    src = Path(args.source)
    if args.source.isdigit():
        run_stream(clf, args.source, args)
    elif src.is_dir():
        run_images(clf, list_images(src), args)
    elif src.is_file() and src.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        run_images(clf, [src], args)
    elif src.is_file():
        run_stream(clf, args.source, args)  # treat as a video file
    else:
        sys.exit(f"Source not found: {args.source}")


if __name__ == "__main__":
    main()
