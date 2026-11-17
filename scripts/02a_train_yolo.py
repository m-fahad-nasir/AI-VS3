"""
Step 2a - Train the leaf detector (manuscript Section 4.1, Table 3).

Trains an Ultralytics YOLOv11 model on the leaf bounding-box dataset and
reports mAP@50, mAP@50-95, precision and recall on the validation split.

Example
-------
python scripts/02a_train_yolo.py --config configs/yolo.yaml
python scripts/02a_train_yolo.py --config configs/yolo.yaml --model yolo11m.pt --name yolo11m_leaf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/yolo.yaml")
    ap.add_argument("--model", help="override the model weights in the config (e.g. yolo11n.pt)")
    ap.add_argument("--name", help="override the run name in the config")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.model:
        cfg["model"] = args.model
    if args.name:
        cfg["name"] = args.name

    model = YOLO(cfg.pop("model"))
    model.train(**cfg)

    metrics = model.val(data=cfg["data"], imgsz=cfg.get("imgsz", 640), device=cfg.get("device"))
    summary = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    out = Path(cfg.get("project", "outputs/detection")) / cfg.get("name", "train") / "val_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
