"""
Step 2b - Detect leaves and crop the primary leaf of every image.

The crop (box + 10 % padding) is saved under the original file name, so
step 3 can run on full images (``--input full``) or crops (``--input crop``)
with identical keys.

Boxes come either from a trained detector (``--weights``) or from existing
YOLO label files (``--labels``).

Examples
--------
python scripts/02b_detect_and_crop.py --images data/images --out data/crops \
       --weights outputs/detection/yolo11l_leaf/weights/best.pt
python scripts/02b_detect_and_crop.py --images data/images --out data/crops --labels data/yolo/labels
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aivs3.detection import open_upright, read_yolo_labels, save_crop, select_primary  # noqa: E402
from aivs3.vlm import list_images  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder of full images")
    ap.add_argument("--out", required=True, help="output folder for crops")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights", help="trained Ultralytics weights (.pt)")
    src.add_argument("--labels", help="folder of YOLO .txt label files")
    ap.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--padding", type=float, default=0.10, help="padding as a fraction of box size")
    ap.add_argument("--primary", choices=["confidence", "largest", "central"], default="confidence",
                    help="rule for choosing the primary leaf when several are detected")
    args = ap.parse_args()

    # Label files carry no confidence, so the largest box is taken as the primary leaf.
    primary = "largest" if args.labels and args.primary == "confidence" else args.primary
    model = None
    if args.weights:
        from ultralytics import YOLO
        model = YOLO(args.weights)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in tqdm(list_images(Path(args.images)), desc="cropping"):
        img = open_upright(path)
        if model is not None:
            res = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            boxes = [tuple(b) for b in res.boxes.xyxy.cpu().tolist()]
            scores = res.boxes.conf.cpu().tolist()
        else:
            label = Path(args.labels) / f"{path.stem}.txt"
            boxes = read_yolo_labels(label, img.width, img.height) if label.exists() else []
            scores = None

        idx = select_primary(boxes, scores, img.width, img.height, primary)
        if idx is None:
            manifest.append({"file": path.name, "boxes": 0, "status": "no leaf found"})
            continue
        save_crop(img, boxes[idx], out_dir / path.name, padding=args.padding)
        manifest.append({"file": path.name, "boxes": len(boxes), "status": "cropped",
                         "x1": round(boxes[idx][0]), "y1": round(boxes[idx][1]),
                         "x2": round(boxes[idx][2]), "y2": round(boxes[idx][3]),
                         "score": round(scores[idx], 4) if scores else ""})

    keys = ["file", "boxes", "status", "x1", "y1", "x2", "y2", "score"]
    with open(out_dir / "crop_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(manifest)
    n_ok = sum(m["status"] == "cropped" for m in manifest)
    print(f"Cropped {n_ok}/{len(manifest)} images into {out_dir}")


if __name__ == "__main__":
    main()
