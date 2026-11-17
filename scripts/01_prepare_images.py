"""
Step 1 - Organise the raw VS3 scouting images.

The VS3 protocol captures 10 images per plant per session, in this order:
one ID-tag image, then Low-A..C, Mid-A..C and High-A..C (Section 3).

Sub-commands
------------
rename       Rename each plant folder's images by capture time to the VS3 scheme
             <session>_C1-001 ID.jpg, <session>_C1-001-Low-A.jpg, ... -High-C.jpg
collect      Copy the renamed images of all plant folders into one session folder
             and move the ID-tag images to a separate folder.
resize       Write half-resolution copies (Lanczos) to an output folder.

Examples
--------
python scripts/01_prepare_images.py rename  --root "raw/Session_03_Feb" --session Date_03_Feb_2026 --dry-run
python scripts/01_prepare_images.py collect --root "raw/Session_03_Feb" --out data/images_full_res --id-dir data/id_images
python scripts/01_prepare_images.py resize  --src data/images_full_res --out data/images --scale 0.5
"""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
CAPTURE_TIME = re.compile(r"(\d{8}_\d{6})")          # camera names: 20260226_114756.jpg
VS3_ORDER = ["ID", "Low-A", "Low-B", "Low-C", "Mid-A", "Mid-B", "Mid-C", "High-A", "High-B", "High-C"]


def images_in(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def capture_time(path: Path) -> datetime:
    m = CAPTURE_TIME.search(path.stem)
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S") if m else datetime.max


def plant_folders(root: Path):
    """Plant folders are named 'Plant C1-001', 'Plant C1-002', ..."""
    return sorted(p for p in root.iterdir() if p.is_dir() and re.search(r"C\d+-\d{3}", p.name))


def cmd_rename(args) -> None:
    root = Path(args.root)
    flagged, planned = [], []
    for folder in plant_folders(root):
        plant = re.search(r"(C\d+-\d{3})", folder.name).group(1)
        imgs = sorted(images_in(folder), key=capture_time)
        if len(imgs) != len(VS3_ORDER):
            flagged.append((folder.name, len(imgs)))
            continue
        for src, tag in zip(imgs, VS3_ORDER):
            stem = f"{args.session}_{plant} ID" if tag == "ID" else f"{args.session}_{plant}-{tag}"
            planned.append((src, src.with_name(stem + src.suffix.lower())))

    for src, dst in planned:
        print(f"{src.parent.name}/{src.name}  ->  {dst.name}")
    if flagged:
        print("\nFolders without exactly 10 images (left unchanged):")
        for name, n in flagged:
            print(f"  {name}: {n} images")
    if args.dry_run:
        print(f"\nDry run: {len(planned)} files would be renamed.")
        return

    # Two-phase rename so that no target name collides with a source name.
    temps = []
    for src, dst in planned:
        tmp = src.with_name(".__tmp__" + src.name)
        src.rename(tmp)
        temps.append((tmp, dst))
    for tmp, dst in temps:
        tmp.rename(dst)
    print(f"\nRenamed {len(temps)} files.")


def cmd_collect(args) -> None:
    out, id_dir = Path(args.out), Path(args.id_dir)
    out.mkdir(parents=True, exist_ok=True)
    id_dir.mkdir(parents=True, exist_ok=True)
    n_img = n_id = 0
    for folder in plant_folders(Path(args.root)):
        for img in sorted(images_in(folder)):
            if img.stem.endswith(" ID"):
                shutil.copy2(img, id_dir / img.name)
                n_id += 1
            else:
                shutil.copy2(img, out / img.name)
                n_img += 1
    print(f"Copied {n_img} scouting images to {out} and {n_id} ID-tag images to {id_dir}.")


def cmd_resize(args) -> None:
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in sorted(images_in(src)):
        with Image.open(img_path) as img:
            size = (max(1, int(img.width * args.scale)), max(1, int(img.height * args.scale)))
            resized = img.resize(size, Image.Resampling.LANCZOS)
            exif = img.info.get("exif")
            kwargs = {"quality": 95} if img_path.suffix.lower() in {".jpg", ".jpeg"} else {}
            if exif:
                kwargs["exif"] = exif
            resized.save(out / img_path.name, **kwargs)
        count += 1
    print(f"Resized {count} images (scale {args.scale}) into {out}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("rename", help="rename images to the VS3 naming scheme by capture time")
    p.add_argument("--root", required=True, help="session folder containing 'Plant C1-XXX' folders")
    p.add_argument("--session", required=True, help="session prefix, e.g. Date_03_Feb_2026")
    p.add_argument("--dry-run", action="store_true", help="only print the planned renames")
    p.set_defaults(func=cmd_rename)

    p = sub.add_parser("collect", help="gather renamed images into one folder")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--id-dir", required=True)
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("resize", help="write resized copies")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, default=0.5)
    p.set_defaults(func=cmd_resize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
