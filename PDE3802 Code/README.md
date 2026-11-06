# OfficeBot Part B – office item classifier

Recognises one office item per image / camera frame and outputs the predicted class and confidence.
Ten classes (as in Table 1 of the report): `mug, bottle, book, notebook, stapler, mouse, phone, keyboard, chair, bin`.

Approach: fine-tune an ImageNet-pretrained YOLO classification model (YOLO11n-cls and/or YOLO26n-cls),
calibrate its confidence with temperature scaling, export to NCNN/ONNX for the Raspberry Pi 4.
Small helpers (Wilson interval, ECE, temperature fit, confusion matrix, macro-F1) are plain NumPy, so there is no
scikit-learn/scipy dependency.

| File | Purpose |
|---|---|
| `split_dataset.py` | session-based train/val/test split (no near-duplicate leakage) |
| `train.py` | fine-tune `yolo11n-cls` / `yolo26n-cls`, optional NCNN/ONNX export |
| `evaluate.py` | accuracy, macro-F1, per-class P/R with Wilson CIs, confusion matrix, temperature scaling, ECE, FPS |
| `predict.py` | **the program**: image file(s), folder, video file or live camera -> class + confidence |
| `office_cls.py` | shared model wrapper and helpers |

## Workflow

```bash
pip install -r requirements.txt

# 1. Photos: raw/<class>/<session>_<number>.jpg   e.g. raw/mug/s01_0001.jpg   (aim for 200/class, min 150)
python split_dataset.py --raw raw/dataset --out data

# 2. Train (try both, keep the better macro-F1 that meets the FPS target)
python train.py --data data --model yolo11n-cls.pt --imgsz 224 --name yolo11 --export ncnn
python train.py --data data --model yolo26n-cls.pt --imgsz 224 --name yolo26 --export ncnn

# 3. Evaluate on the held-out test set (fits temperature on val, writes calibration.json next to the weights)
python evaluate.py --weights runs/classify/runs/yolo11/weights/best.pt --val data/val --test data/test --imgsz 224 --bench

# 4. Use it
python predict.py --weights runs/classify/runs/yolo11/weights/best.pt --source photo.jpg
python predict.py --weights runs/classify/runs/yolo11/weights/best.pt --source raw/dataset/mug --csv results.csv
python predict.py --weights runs/classify/runs/yolo11/weights/best.pt --source 0                # live webcam
```

On the Pi 4 copy the `best_ncnn_model/` folder, `calibration.json` and these scripts, then run
`evaluate.py --bench` (once with `--imgsz 224`, once with `--imgsz 320`; train at the size you deploy)
to fill in the FPS placeholders. `predict.py --no-display` works over SSH.

## Notes
* **Calibration:** Ultralytics only returns softmax probabilities. Because log(p) = logits - constant per sample and
  the constant cancels in softmax, `softmax(log(p)/T)` equals temperature scaling on the logits. `T` is fitted on
  the validation set only, never on test.
* **Augmentation:** Ultralytics' classification pipeline does not take the detection-style `degrees` argument, so small
  rotations come from RandAugment; flips, scale and brightness/colour jitter are set explicitly in `train.py`.
* **Early stopping** uses Ultralytics' fitness (top-1 accuracy), not macro-F1; with balanced classes they track closely.
  Compare final macro-F1 from `evaluate.py`.
* `--min-conf 0.9` makes `predict.py` print `unsure` for low-confidence predictions (the report's "unsure -> alert" rule).
* Live mode averages probabilities over frames (`--smooth`), and `--center-crop 0.7` enforces the one-item-at-a-time
  assumption by classifying only the central square.
