"""
Step 6 - Compute the hierarchical Health Index (manuscript Section 4.3).

Scores every plant-session and writes:
  <out>/<name>_Threat_Scored.xlsx    image, level and plant tables + coefficients
  <out>/<name>_Health_Index.csv      one row per plant, one column per session

Examples
--------
# From the agronomist annotations (Sections 5.6 and 5.7)
python scripts/06_score_health_index.py --input data/Autonomous_Scouting_Dataset.xlsx \
       --sheets Week_01 Week_02 Week_03 Week_04 --source annotation --name Agronomist

# From one model's predictions in the results workbook
python scripts/06_score_health_index.py --input outputs/VLM_Results.xlsx \
       --sheets GPT_5.2 --source predicted --name GPT_5.2

# From every model sheet of the results workbook
python scripts/06_score_health_index.py --input outputs/VLM_Results.xlsx --source predicted --all-sheets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aivs3.temporal import trajectory_table  # noqa: E402
from aivs3.tomato_threat_scoring import score_workbook  # noqa: E402

SCOUTING_SESSIONS = ["Date_03_Feb_2026", "Date_13_Feb_2026", "Date_17_Feb_2026", "Date_26_Feb_2026"]


def run(input_file: Path, sheets, source: str, name: str, out_dir: Path, sessions) -> pd.DataFrame:
    _, _, plants = score_workbook(input_file, sheets, source=source,
                                  output_file=out_dir / f"{name}_Threat_Scored.xlsx", sessions=sessions)
    wide = trajectory_table(plants, sessions=sessions)
    wide.columns = [f"{c} Health Index" for c in wide.columns]
    wide.reset_index().rename(columns={"Plant_Stem_Number": "Plant"}).to_csv(
        out_dir / f"{name}_Health_Index.csv", index=False)
    return plants


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="annotation workbook or VLM results workbook")
    ap.add_argument("--sheets", nargs="*", help="sheets to score together")
    ap.add_argument("--all-sheets", action="store_true", help="score every model sheet separately")
    ap.add_argument("--source", choices=["annotation", "gt", "predicted"], default="annotation",
                    help="which label columns to score")
    ap.add_argument("--name", default="Agronomist", help="prefix of the output files")
    ap.add_argument("--sessions", nargs="*", default=SCOUTING_SESSIONS,
                    help="session tags to keep (default: the four scouting dates)")
    ap.add_argument("--out", default="outputs/health_index")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_file = Path(args.input)

    if args.all_sheets:
        sheets = [s for s in pd.ExcelFile(input_file).sheet_names if s != "Master_Template"]
        rows = []
        for sheet in sheets:
            plants = run(input_file, [sheet], args.source, sheet, out_dir, args.sessions)
            rows.append({"Model": sheet, "Plant-sessions": len(plants),
                         "Mean_HI": plants["Health_Index"].mean(),
                         **plants["Alert_Level"].value_counts().to_dict()})
            print(f"{sheet:<16} plant-sessions={len(plants):>4}  mean HI={plants['Health_Index'].mean():.3f}")
        pd.DataFrame(rows).to_csv(out_dir / "model_overview.csv", index=False)
    else:
        if not args.sheets:
            sys.exit("Give --sheets or --all-sheets")
        plants = run(input_file, args.sheets, args.source, args.name, out_dir, args.sessions)
        stats = plants.groupby("Session")["Health_Index"].agg(["count", "mean", "median"])
        print(stats.reindex(args.sessions).round(3).to_string())
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
