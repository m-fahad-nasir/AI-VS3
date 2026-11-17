"""
Step 4 - Build the results workbook: annotations next to every model's predictions.

Reads the session sheets (Week_01 ... Week_04) of the annotation workbook and
the JSONL outputs of step 3, and writes one sheet per model with
``<field> (GT)`` and ``<field> (Predicted)`` columns.

Example
-------
python scripts/04_build_results_workbook.py \
       --annotations data/Autonomous_Scouting_Dataset.xlsx \
       --jsonl outputs/vlm_raw/crop --out outputs/VLM_Results.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aivs3.workbook import load_annotations, write_results_workbook  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default="data/Autonomous_Scouting_Dataset.xlsx")
    ap.add_argument("--sheets", nargs="*", help="annotation sheets (default: every 'Week_*' sheet)")
    ap.add_argument("--jsonl", default="outputs/vlm_raw/crop", help="folder with <model>.jsonl files")
    ap.add_argument("--models", nargs="*", help="model tags (default: all in configs/models.yaml)")
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--out", default="outputs/VLM_Results.xlsx")
    args = ap.parse_args()

    models = args.models or list(yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["models"])
    annotations = load_annotations(Path(args.annotations), args.sheets)
    answered = write_results_workbook(annotations, Path(args.jsonl), models, Path(args.out))

    print(f"Annotated images: {len(annotations)}")
    for model, n in answered.items():
        print(f"  {model:<16} predictions: {n}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
