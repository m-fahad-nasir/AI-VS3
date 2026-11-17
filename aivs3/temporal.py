"""
Temporal Health Index trajectories (manuscript Sections 4.4 and 5.7).

A trajectory is the sequence of plant-level Health Index values over the
scouting sessions. Each plant receives a trend class from
Delta = HI(last) - HI(first):

    strong decline       Delta <= -0.30
    decline              -0.30 < Delta <= -0.10
    stable               |Delta| < 0.10
    improvement          0.10 <= Delta < 0.30
    strong improvement   Delta >= 0.30

A trajectory is additionally flagged as volatile if it contains both a rise
and a fall of at least 0.20 between consecutive sessions.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

TREND_ORDER = ["Strong decline", "Decline", "Stable", "Improvement", "Strong improvement"]
TREND_RANGES = {
    "Strong decline": "<= -0.30",
    "Decline": "(-0.30, -0.10]",
    "Stable": "(-0.10, 0.10)",
    "Improvement": "[0.10, 0.30)",
    "Strong improvement": ">= 0.30",
}
VOLATILITY_STEP = 0.20


def session_sort_key(session: str):
    """Chronological key for ``Date_03_Feb_2026``-style session tags."""
    try:
        return datetime.strptime(str(session).replace("Date_", ""), "%d_%b_%Y")
    except ValueError:
        return str(session)


def trajectory_table(plant_scores: pd.DataFrame, value: str = "Health_Index",
                     sessions: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Pivot plant scores into one row per plant and one column per session."""
    wide = plant_scores.pivot_table(index="Plant_Stem_Number", columns="Session",
                                    values=value, aggfunc="first")
    cols = list(sessions) if sessions else sorted(wide.columns, key=session_sort_key)
    wide = wide.reindex(columns=cols)
    wide.columns = [f"Week {i}" for i in range(1, len(cols) + 1)]
    return wide.sort_index()


def trend_class(delta: float) -> str:
    if pd.isna(delta):
        return "Incomplete"
    if delta <= -0.30:
        return "Strong decline"
    if delta <= -0.10:
        return "Decline"
    if delta < 0.10:
        return "Stable"
    if delta < 0.30:
        return "Improvement"
    return "Strong improvement"


def is_volatile(values: Sequence[float], step: float = VOLATILITY_STEP) -> bool:
    steps = np.diff(np.asarray(values, dtype=float))
    steps = steps[~np.isnan(steps)]
    return bool((steps >= step).any() and (steps <= -step).any())


def classify_trajectories(wide: pd.DataFrame) -> pd.DataFrame:
    """Add Delta, trend class and volatility to a trajectory table."""
    out = wide.copy()
    first, last = wide.columns[0], wide.columns[-1]
    out["Delta"] = wide[last] - wide[first]
    out["Trend"] = out["Delta"].apply(trend_class)
    out["Volatile"] = [is_volatile(row) for row in wide.to_numpy()]
    out["Trend_Label"] = out["Trend"] + np.where(out["Volatile"], " (volatile)", "")
    return out


def trend_distribution(classified: pd.DataFrame) -> pd.DataFrame:
    """Manuscript Table 5: plants per trend class and number of volatile trajectories."""
    n = len(classified)
    rows = []
    for cls in TREND_ORDER:
        sub = classified[classified["Trend"] == cls]
        rows.append({"Trend class": cls, "Delta range": TREND_RANGES[cls], "Plants": len(sub),
                     "%": round(100 * len(sub) / n, 1) if n else np.nan,
                     "Volatile": int(sub["Volatile"].sum())})
    return pd.DataFrame(rows)


def session_statistics(wide: pd.DataFrame) -> pd.DataFrame:
    """Mean, median and spread of the Health Index per session (Figure 7)."""
    return pd.DataFrame({
        "Session": wide.columns,
        "Plants": wide.notna().sum().to_numpy(),
        "Mean_HI": wide.mean().to_numpy(),
        "Median_HI": wide.median().to_numpy(),
        "Std_HI": wide.std().to_numpy(),
        "Q1_HI": wide.quantile(0.25).to_numpy(),
        "Q3_HI": wide.quantile(0.75).to_numpy(),
    })


# =============================================================================
# Figures
# =============================================================================
def plot_population_trend(wide: pd.DataFrame, path: Path) -> None:
    """Figure 7: distribution, mean and median Health Index per session."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = session_statistics(wide)
    x = np.arange(1, len(wide.columns) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([wide[c].dropna() for c in wide.columns], positions=x, widths=0.45,
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#d9ead3", edgecolor="#6a8f5f"),
               medianprops=dict(color="#6a8f5f"))
    ax.plot(x, stats["Mean_HI"], "o-", color="#1f4e5f", label="Mean")
    ax.plot(x, stats["Median_HI"], "s--", color="#c0392b", label="Median")
    ax.set_xticks(x, wide.columns)
    ax.set_ylabel("Health Index")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_trajectory_heatmap(wide: pd.DataFrame, path: Path) -> None:
    """Figure 8: Health Index per plant (columns) and session (rows)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = wide.T
    fig, ax = plt.subplots(figsize=(max(10, 0.32 * data.shape[1]), 3.2))
    im = ax.imshow(data.to_numpy(dtype=float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(data.shape[0]), data.index)
    labels: List[str] = [str(p).replace("Plant ", "") for p in data.columns]
    ax.set_xticks(range(data.shape[1]), labels, rotation=90, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Health Index")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
