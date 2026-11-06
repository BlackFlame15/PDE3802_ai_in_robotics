#!/usr/bin/env python3
"""
Produces: accuracy, macro-F1, per-class precision/recall with Wilson 95% intervals,
confusion matrix (CSV + PNG), calibration (temperature fitted on VAL, ECE on TEST,
reliability diagram) and optional single-image latency/FPS. The fitted temperature
is written to calibration.json next to the weights; predict.py picks it up.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from office_cls import OfficeClassifier, apply_temperature, list_images, read_image


# ---------- small, dependency-free metric helpers ----------
def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def confusion(y, pred, n):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return cm


def nll(probs, y):
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    hit = pred == y
    edges = np.linspace(0, 1, bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
            rows.append((lo, hi, int(m.sum()), float(conf[m].mean()), float(hit[m].mean())))
    return float(total), rows


def fit_temperature(probs, y):
    grid = np.exp(np.linspace(np.log(0.2), np.log(10), 400))
    return float(min(grid, key=lambda T: nll(apply_temperature(probs, T), y)))


def per_class(cm, names):
    out = []
    for i, name in enumerate(names):
        tp, support, predicted = cm[i, i], cm[i].sum(), cm[:, i].sum()
        prec = tp / predicted if predicted else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append({"class": name, "support": int(support), "precision": prec, "recall": rec, "f1": f1,
                    "precision_ci": wilson(int(tp), int(predicted)), "recall_ci": wilson(int(tp), int(support))})
    return out


# ---------- data ----------
def collect(clf, folder):
    idx = {n: i for i, n in enumerate(clf.names)}
    probs, ys, paths = [], [], []
    for cdir in sorted(d for d in Path(folder).iterdir() if d.is_dir()):
        if cdir.name not in idx:
            print(f"warning: folder {cdir.name!r} is not a model class, skipped")
            continue
        for p in list_images(cdir):
            img = read_image(p)
            if img is None:
                continue
            probs.append(clf.probs(img, calibrated=False))
            ys.append(idx[cdir.name])
            paths.append(str(p))
    return np.array(probs), np.array(ys), paths


def bench(clf, paths, runs=100):
    imgs = [read_image(p) for p in paths[:20]]
    for i in range(10):  # warm-up
        clf.probs(imgs[i % len(imgs)])
    t = []
    for i in range(runs):
        t0 = time.perf_counter()
        clf.probs(imgs[i % len(imgs)])
        t.append(time.perf_counter() - t0)
    med = float(np.median(t))
    return {"median_ms": med * 1000, "p95_ms": float(np.percentile(t, 95)) * 1000, "fps": 1 / med}


# ---------- plots (optional) ----------
def save_plots(out, names, cm, bins):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping PNG plots")
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.bar([(lo + hi) / 2 for lo, hi, *_ in bins], [b[4] for b in bins], width=1 / 15, alpha=0.7, edgecolor="k")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram (test, calibrated)")
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--test", required=True, help="held-out test folder (class sub-folders)")
    ap.add_argument("--val", help="validation folder, used ONLY to fit the temperature")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--out", default="eval")
    ap.add_argument("--bench", action="store_true", help="measure per-frame latency / FPS on this machine")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clf = OfficeClassifier(args.weights, args.imgsz, temperature=1.0, device=args.device)
    names = clf.names

    T = 1.0
    if args.val:
        vp, vy, _ = collect(clf, args.val)
        T = fit_temperature(vp, vy)
        (Path(args.weights).parent / "calibration.json").write_text(json.dumps({"temperature": T, "imgsz": args.imgsz}))
        print(f"Fitted temperature on {len(vy)} val images: T = {T:.3f}")

    probs, y, paths = collect(clf, args.test)
    pred = probs.argmax(1)
    cm = confusion(y, pred, len(names))
    rows = per_class(cm, names)
    acc = float((pred == y).mean())
    lo, hi = wilson(int((pred == y).sum()), len(y))
    macro_f1 = float(np.mean([r["f1"] for r in rows]))
    cal = apply_temperature(probs, T)
    ece_raw, _ = ece(probs, y)
    ece_cal, bins = ece(cal, y)

    print(f"\nTest images: {len(y)}  imgsz: {args.imgsz}")
    print(f"Accuracy : {acc:.3f}  (Wilson 95% CI {lo:.3f}-{hi:.3f})")
    print(f"Macro-F1 : {macro_f1:.3f}")
    print(f"ECE      : {ece_raw:.3f} raw -> {ece_cal:.3f} after temperature scaling (T={T:.2f})")
    print(f"NLL      : {nll(probs, y):.3f} raw -> {nll(cal, y):.3f} calibrated\n")
    print(f"{'class':<10}{'n':>4}  {'precision [95% CI]':<24}{'recall [95% CI]':<24}{'F1':>5}")
    for r in rows:
        p_ci, r_ci = r["precision_ci"], r["recall_ci"]
        print(f"{r['class']:<10}{r['support']:>4}  {r['precision']:.2f} [{p_ci[0]:.2f}-{p_ci[1]:.2f}]".ljust(40)
              + f"{r['recall']:.2f} [{r_ci[0]:.2f}-{r_ci[1]:.2f}]".ljust(24) + f"{r['f1']:>5.2f}")
    worst = sorted(((cm[i, j], names[i], names[j]) for i in range(len(names)) for j in range(len(names)) if i != j),
                   reverse=True)[:5]
    print("\nTop confusions (true -> predicted):", ", ".join(f"{a}->{b} x{n}" for n, a, b in worst if n))

    result = {"imgsz": args.imgsz, "n_test": len(y), "accuracy": acc, "accuracy_ci": [lo, hi], "macro_f1": macro_f1,
              "temperature": T, "ece_raw": ece_raw, "ece_calibrated": ece_cal, "per_class": rows}
    if args.bench:
        result["speed"] = bench(clf, paths)
        s = result["speed"]
        print(f"\nSpeed @ {args.imgsz}px: median {s['median_ms']:.1f} ms, p95 {s['p95_ms']:.1f} ms -> {s['fps']:.1f} FPS "
              "(model + preprocessing; camera capture not included)")
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    with open(out / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + names)
        for n, row in zip(names, cm):
            w.writerow([n] + list(row))
    save_plots(out, names, cm, bins)
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
