"""
Step 5 - Evaluate every VLM against the agronomist annotations (Table 4).

Reports per model: accuracy for health status, disease category, disease,
severity and confidence (plus macro-F1 and Cohen's kappa), and BLEU-4,
ROUGE-L and TF-IDF cosine for the free-text fields. Also writes per-image
correctness and confusion matrices.

Example
-------
python scripts/05_evaluate_vlms.py --input outputs/VLM_Results.xlsx --out outputs/evaluation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aivs3.evaluation import confusion_table, evaluate_workbook, table4  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="outputs/VLM_Results.xlsx")
    ap.add_argument("--out", default="outputs/evaluation")
    ap.add_argument("--text-field", choices=["reasoning", "visible_symptoms"], default="reasoning",
                    help="free-text field reported in the Table 4 layout")
    ap.add_argument("--header-row", type=int, default=1, help="0-based header row of each model sheet")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "confusion").mkdir(parents=True, exist_ok=True)
    summary, per_image = evaluate_workbook(Path(args.input), header_row=args.header_row)
    t4 = table4(summary, args.text_field)

    summary.to_csv(out / "model_summary.csv", index=False)
    t4.to_csv(out / "table4.csv", index=False)
    with pd.ExcelWriter(out / "evaluation.xlsx", engine="openpyxl") as writer:
        t4.to_excel(writer, sheet_name="Table4", index=False)
        summary.to_excel(writer, sheet_name="All_Metrics", index=False)
        for model, df in per_image.items():
            df.to_excel(writer, sheet_name=f"img_{model}"[:31], index=False)
    for model, df in per_image.items():
        for key in ("health", "category", "disease", "severity"):
            confusion_table(df[f"{key}_gt"], df[f"{key}_pred"]).to_csv(out / "confusion" / f"{model}_{key}.csv")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t4.to_string(index=False))
    print(f"\nSaved tables to {out}")


if __name__ == "__main__":
    main()
