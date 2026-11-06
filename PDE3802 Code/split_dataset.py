#!/usr/bin/env python3
import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from office_cls import list_images

SPLITS = ("train", "val", "test")
SESSION_RE = re.compile(r"^s\d+_")  # e.g. s01_0001.jpg -> session "s01"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--test-sessions", nargs="*", default=[], help="force these sessions into test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    classes = sorted(d.name for d in raw.iterdir() if d.is_dir())
    if not classes:
        sys.exit(f"No class folders under {raw}/")
    # Common mistake: pointing at a wrapper folder (e.g. raw/ containing only dataset/<class>/)
    if len(classes) == 1 and any(d.is_dir() for d in (raw / classes[0]).iterdir()):
        sys.exit(f"Only one folder under {raw}/ ({classes[0]!r}) and it contains sub-folders; "
                 f"did you mean --raw {raw / classes[0]} ?")
    if len(classes) < 2:
        sys.exit(f"Need >= 2 class folders under {raw}/, found {classes}")

    images = {c: list_images(raw / c) for c in classes}
    all_imgs = [p for ps in images.values() for p in ps]
    session_mode = bool(all_imgs) and all(SESSION_RE.match(p.stem) for p in all_imgs)

    # assignment: path -> split
    split_of = {}
    if session_mode:
        by_session = defaultdict(list)  # session -> [path]
        for p in all_imgs:
            by_session[p.stem.split("_")[0]].append(p)
        if len(by_session) < 3:
            sys.exit(f"Need >= 3 capture sessions to make 3 splits, found {len(by_session)}")

        target = dict(zip(SPLITS, (r * len(all_imgs) for r in args.ratios)))
        count = dict.fromkeys(target, 0)
        assign = {}
        for s in args.test_sessions:
            if s not in by_session:
                sys.exit(f"Unknown session {s!r}; found {sorted(by_session)}")
            assign[s] = "test"
            count["test"] += len(by_session[s])

        rest = [s for s in by_session if s not in assign]
        random.Random(args.seed).shuffle(rest)
        for s in sorted(rest, key=lambda s: -len(by_session[s])):  # biggest first, to the split with most room
            split = max(target, key=lambda k: target[k] - count[k])
            assign[s] = split
            count[split] += len(by_session[s])
        for s, ps in by_session.items():
            for p in ps:
                split_of[p] = assign[s]
    else:
        # Filenames don't follow sNN_xxxx: no session info, so do a stratified random
        # split (same train/val/test ratios inside every class).
        if args.test_sessions:
            sys.exit("--test-sessions needs files named like s01_0001.jpg")
        print("No sNN_ session prefixes found: using a stratified random split per class.\n"
              "(Near-duplicate photos may end up in different splits.)\n")
        rng = random.Random(args.seed)
        tr, va, _ = args.ratios
        for c in classes:
            ps = images[c][:]
            rng.shuffle(ps)
            n_tr, n_va = round(tr * len(ps)), round(va * len(ps))
            for i, p in enumerate(ps):
                split_of[p] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"

    if out.exists():
        shutil.rmtree(out)
    per = defaultdict(lambda: defaultdict(int))  # class -> split -> n
    for c in classes:
        for p in images[c]:
            dst = out / split_of[p] / c
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / p.name)
            per[c][split_of[p]] += 1

    print(f"{'class':<12}{'train':>7}{'val':>6}{'test':>6}{'total':>7}")
    for c in classes:
        n = per[c]
        tot = sum(n.values())
        flag = ""
        if min(n["train"], n["val"], n["test"]) == 0:
            flag = "  <-- EMPTY SPLIT: add photos from more sessions"
        elif tot < 150:
            flag = "  <-- below the 150-image minimum"
        print(f"{c:<12}{n['train']:>7}{n['val']:>6}{n['test']:>6}{tot:>7}{flag}")
    if session_mode:
        print("\nsessions:", {k: sorted(s for s, v in assign.items() if v == k) for k in SPLITS})


if __name__ == "__main__":
    main()
