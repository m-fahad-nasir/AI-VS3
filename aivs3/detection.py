"""
Leaf detection and crop extraction (manuscript Section 4.1).

A detector (Ultralytics YOLOv11) locates the leaves in each image. The primary
leaf is cropped with a padding margin and saved under the original file name,
so VLM outputs on crops and on full images share the same keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageOps

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


def pad_box(box: Box, width: int, height: int, padding: float = 0.10) -> Tuple[int, int, int, int]:
    """Grow a box by ``padding`` of its size on every side, clipped to the image."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return (max(0, int(round(x1 - dx))), max(0, int(round(y1 - dy))),
            min(width, int(round(x2 + dx))), min(height, int(round(y2 + dy))))


def select_primary(boxes: Sequence[Box], scores: Optional[Sequence[float]], width: int, height: int,
                   strategy: str = "confidence") -> Optional[int]:
    """Index of the primary leaf: highest confidence, largest area, or closest to the centre."""
    if len(boxes) == 0:
        return None
    if strategy == "confidence" and scores is not None:
        return int(max(range(len(boxes)), key=lambda i: scores[i]))
    if strategy == "largest":
        return int(max(range(len(boxes)), key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])))
    cx, cy = width / 2, height / 2
    return int(min(range(len(boxes)),
                   key=lambda i: ((boxes[i][0] + boxes[i][2]) / 2 - cx) ** 2 + ((boxes[i][1] + boxes[i][3]) / 2 - cy) ** 2))


def read_yolo_labels(label_path: Path, width: int, height: int) -> List[Box]:
    """Read a YOLO ``.txt`` label file (class xc yc w h, normalised) as pixel boxes."""
    boxes = []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        xc, yc, w, h = (float(v) for v in parts[1:5])
        boxes.append(((xc - w / 2) * width, (yc - h / 2) * height,
                      (xc + w / 2) * width, (yc + h / 2) * height))
    return boxes


def open_upright(path: Path) -> Image.Image:
    """Open an image in its EXIF display orientation (the frame used for labelling)."""
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def save_crop(img: Image.Image, box: Box, out_path: Path, padding: float = 0.10) -> None:
    crop = img.crop(pad_box(box, img.width, img.height, padding))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        crop.save(out_path, quality=95)
    else:
        crop.save(out_path)
