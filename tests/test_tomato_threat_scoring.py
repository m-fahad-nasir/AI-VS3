import pandas as pd
import pytest

from aivs3 import tomato_threat_scoring as hi

SESSION = "Date_03_Feb_2026"
VIEWS = ["A", "B", "C"]
LEVELS = {"High": "High", "Middle": "Mid", "Low": "Low"}


def plant_frame(overrides=None):
    """One plant-session, 3 levels x 3 images, healthy unless overridden.

    overrides: {(level, view): (category, disease, severity)}
    """
    overrides = overrides or {}
    rows = []
    for level, short in LEVELS.items():
        for view in VIEWS:
            cat, dis, sev = overrides.get((level, view), (None, None, None))
            rows.append({
                "Plant_Stem_Number": "Plant C1-001" if not rows else None,
                "Plant Images Level": level,
                "File Name": f"{SESSION}_C1-001-{short}-{view}.jpg",
                "Health Status": "Not Healthy" if dis else "Healthy",
                "Disease Category": cat,
                "Disease": dis,
                "Severity": sev,
                "Confidence": "High",
            })
    return hi.prepare_frame(pd.DataFrame(rows))


def plant_row(df):
    _, _, plants = hi.score_frame(df)
    assert len(plants) == 1
    return plants.iloc[0]


# --- agreement bands ---------------------------------------------------------
@pytest.mark.parametrize("labels, expected", [
    (["healthy", "healthy", "healthy"], 1.00),
    (["a", "a", "a"], 1.00),
    (["a", "a", "b"], 0.75),
    (["a", "b", "c"], 0.50),
    (["a", "healthy", "healthy"], 0.25),
])
def test_disease_agreement(labels, expected):
    assert hi.disease_agreement(labels) == expected


@pytest.mark.parametrize("severities, expected", [
    (["Medium", "", ""], 1.00),
    (["Medium", "Medium", "Medium"], 1.00),
    (["Low", "Medium", ""], 0.75),
    (["Medium", "High", "High"], 0.75),
    (["Low", "High", ""], 0.40),
])
def test_severity_agreement(severities, expected):
    assert hi.severity_agreement(severities) == expected


# --- threat coefficients -----------------------------------------------------
@pytest.mark.parametrize("category, disease, expected", [
    ("Viral Leaf Disease", "Tomato yellow leaf curl virus (TYLCV)", 1.00),
    ("Viral Leaf Disease", "Cucumber mosaic virus", 0.85),
    ("Fungal Leaf Disease", "Target spot", 0.75),
    ("Nutrient Disease", "Magnesium deficiency", 0.35),
    ("Abiotic", "Environmental stress", 0.45),
    ("Abiotic", "Mechanical damage", 0.35),
    ("Bacterial Leaf Disease", "Unknown bacterial infection", 0.90),
])
def test_exact_disease_score(category, disease, expected):
    assert hi.exact_disease_score(category, disease, "Not Healthy") == pytest.approx(expected)


def test_healthy_leaf_has_no_threat():
    assert hi.exact_disease_score("Fungal Leaf Disease", "Early blight", "Healthy") == 0.0


# --- full aggregation --------------------------------------------------------
def test_all_healthy_plant():
    row = plant_row(plant_frame())
    assert row["Health_Index"] == pytest.approx(1.0)
    assert row["Alert_Level"] == "Minimal"
    assert row["Plant_Reliability_Q"] == pytest.approx(1.0)
    assert row["Plant_Uncertainty_U"] == pytest.approx(0.0)


def test_nutrient_deficiency_in_top_canopy():
    mg = ("Nutrient Disease", "Magnesium deficiency", "Medium")
    row = plant_row(plant_frame({("High", v): mg for v in VIEWS}))
    # R = 0.35 * 0.60 = 0.21; D = K = 1 -> G = 1.1; T_high = 1.0 * 1.0 * 1.1 * 0.21
    t_high = 1.1 * 0.21
    t_base = 0.75 * t_high + 0.25 * 0.5 * t_high
    assert row["T_High"] == pytest.approx(t_high)
    assert row["Systemic_Boost_B"] == pytest.approx(1.0)
    assert row["Health_Index"] == pytest.approx(1 - t_base)
    assert row["Alert_Level"] == "Low"


def test_systemic_boost_across_levels():
    mg = ("Nutrient Disease", "Magnesium deficiency", "Medium")
    row = plant_row(plant_frame({(lvl, v): mg for lvl in LEVELS for v in VIEWS}))
    assert row["Boosted_Levels_n"] == 3
    assert row["Systemic_Boost_B"] == pytest.approx(1.2)


def test_viral_override_raises_alert_only():
    tylcv = ("Viral Leaf Disease", "Tomato yellow leaf curl virus", "Low")
    row = plant_row(plant_frame({("Low", "A"): tylcv}))
    # R = 0.30, D = 0.25, K = 1 -> A = 0.475, G = 0.9425; alpha = 0.75; M = 1.7; V = 0.65
    t_low = 0.65 * 1.7 * 0.9425 * (0.75 * 0.30 + 0.25 * 0.10)
    t_base = 0.75 * t_low + 0.25 * 0.2 * t_low
    assert row["Health_Index"] == pytest.approx(1 - t_base)
    assert hi.alert_level(t_base) == "Low"
    assert row["Alert_Level"] == "Medium"
    assert row["Review_Flag"] == "Viral review override"


def test_top_fungal_override():
    eb = ("Fungal Leaf Disease", "Early blight", "Low")
    row = plant_row(plant_frame({("High", "A"): eb}))
    assert row["Alert_Level"] == "High"
    assert row["Review_Flag"] == "Top fungal override"


def test_loader_keeps_image_rows_only():
    df = plant_frame()
    extra = pd.DataFrame([{"Plant Images Level": "Healthy", "File Name": 236},
                          {"Plant Images Level": "High", "File Name": "-"}])
    raw = pd.concat([df.drop(columns=["Session", "Level"]), extra], ignore_index=True)
    assert len(hi.prepare_frame(raw)) == 9


def test_session_from_filename():
    assert hi.session_from_filename("Date_26_Feb_2026_C1-018-Mid-A.jpg") == "Date_26_Feb_2026"
    assert hi.session_from_filename("leaf.jpg", "Week_01") == "Week_01"
