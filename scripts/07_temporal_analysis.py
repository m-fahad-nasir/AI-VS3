"""
Step 7 - Temporal Health Index trajectories (manuscript Section 5.7).

Input is the per-plant Health Index table written by step 6. Outputs:
  session_statistics.csv     mean / median HI per session          (Figure 7)
  trajectories.csv           HI per plant and session, Delta, trend, volatility
  trend_distribution.csv     plants per trend class                (Table 5)
  representative.csv         selected plants                       (Table 6)
  fig7_population_trend.png  population-level evolution            (Figure 7)
  fig8_trajectory_heatmap.png plant-level trajectories             (Figure 8)

Example
-------
python scripts/07_temporal_analysis.py --input outputs/health_index/Agronomist_Health_Index.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aivs3.temporal import (classify_trajectories, plot_population_trend,  # noqa: E402
                            plot_trajectory_heatmap, session_statistics, trend_distribution)

REPRESENTATIVE = ["Plant C1-002", "Plant C1-007", "Plant C1-005", "Plant C1-008", "Plant C1-021"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="outputs/health_index/Agronomist_Health_Index.csv")
    ap.add_argument("--out", default="outputs/temporal")
    ap.add_argument("--plants", nargs="*", default=REPRESENTATIVE, help="plants listed in Table 6")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wide = pd.read_csv(args.input).set_index("Plant")
    wide.columns = [c.replace(" Health Index", "") for c in wide.columns]

    stats = session_statistics(wide)
    classified = classify_trajectories(wide)
    table5 = trend_distribution(classified)
    table6 = classified.reindex([p for p in args.plants if p in classified.index])

    stats.to_csv(out / "session_statistics.csv", index=False)
    classified.to_csv(out / "trajectories.csv")
    table5.to_csv(out / "trend_distribution.csv", index=False)
    table6.to_csv(out / "representative.csv")
    plot_population_trend(wide, out / "fig7_population_trend.png")
    plot_trajectory_heatmap(wide, out / "fig8_trajectory_heatmap.png")

    print(stats.round(3).to_string(index=False), "\n")
    print(table5.to_string(index=False), "\n")
    print(table6[list(wide.columns) + ["Delta", "Trend_Label"]].round(3).to_string())
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
