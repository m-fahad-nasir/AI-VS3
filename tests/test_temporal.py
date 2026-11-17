import pandas as pd
import pytest

from aivs3 import temporal


@pytest.mark.parametrize("delta, expected", [
    (-0.55, "Strong decline"),
    (-0.30, "Strong decline"),
    (-0.20, "Decline"),
    (-0.10, "Decline"),
    (0.00, "Stable"),
    (0.10, "Improvement"),
    (0.29, "Improvement"),
    (0.30, "Strong improvement"),
])
def test_trend_class(delta, expected):
    assert temporal.trend_class(delta) == expected


def test_volatility_needs_rise_and_fall():
    assert temporal.is_volatile([0.784, 0.399, 0.917, 0.893])      # C1-005 (Table 6)
    assert temporal.is_volatile([0.591, 0.962, 0.952, 0.623])      # C1-008
    assert not temporal.is_volatile([0.955, 0.682, 0.719, 0.405])  # C1-002: falls only
    assert not temporal.is_volatile([0.942, 0.965, 0.973, 0.945])  # C1-021


def test_table6_examples():
    wide = pd.DataFrame(
        [[0.955, 0.682, 0.719, 0.405], [0.432, 0.524, 0.895, 0.862], [0.784, 0.399, 0.917, 0.893],
         [0.591, 0.962, 0.952, 0.623], [0.942, 0.965, 0.973, 0.945]],
        index=["C1-002", "C1-007", "C1-005", "C1-008", "C1-021"],
        columns=["Week 1", "Week 2", "Week 3", "Week 4"])
    out = temporal.classify_trajectories(wide)
    assert out["Trend_Label"].tolist() == [
        "Strong decline", "Strong improvement", "Improvement (volatile)", "Stable (volatile)", "Stable"]
    dist = temporal.trend_distribution(out).set_index("Trend class")
    assert dist["Plants"].sum() == 5
    assert dist.loc["Stable", "Volatile"] == 1


def test_trajectory_table_orders_sessions_chronologically():
    plants = pd.DataFrame({
        "Plant_Stem_Number": ["P1"] * 3,
        "Session": ["Date_17_Feb_2026", "Date_03_Feb_2026", "Date_13_Feb_2026"],
        "Health_Index": [0.7, 0.9, 0.8],
    })
    wide = temporal.trajectory_table(plants)
    assert wide.loc["P1"].tolist() == [0.9, 0.8, 0.7]
