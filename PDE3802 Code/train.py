#!/usr/bin/env python3
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="yolo11n-cls.pt", help="ImageNet-pretrained classification weights")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10, help="early-stopping patience (epochs)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="office")
    ap.add_argument("--device", default=None)
    ap.add_argument("--export", choices=["onnx", "ncnn"], help="also export the best weights (for the Pi)")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(Path(args.data).resolve()),
        imgsz=args.imgsz, epochs=args.epochs, patience=args.patience, batch=args.batch,
        project=args.project, name=args.name, device=args.device, seed=0, exist_ok=True,
        # augmentation: flips, scale, brightness/contrast-style colour jitter, plus
        # RandAugment (which includes small rotations) for the classification pipeline
        fliplr=0.5, scale=0.3, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, auto_augment="randaugment",
    )
    # Ultralytics decides the real run folder (8.4+ nests it under runs/classify/...)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    if args.export:
        YOLO(str(best)).export(format=args.export, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
