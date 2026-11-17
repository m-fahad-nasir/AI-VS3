"""
AI-VS3 hierarchical Health Index (manuscript Section 4.3, Algorithm 1).

Pipeline
--------
image  : R_i = E_i * S_i                                   (Eq. 5)
level  : T_l = min{1, V_l * M_l * G_l * [a_l*max R + (1-a_l)*mean R]}   (Eq. 6)
         G_l = 0.8 + 0.3 * A_l,  A_l = 0.7 * D_l + 0.3 * K_l           (Eq. 7)
plant  : T_base = 0.75 * max_l T_l + 0.25 * sum_l W_l * T_l          (Eq. 8)
         T_P = min{1, B * T_base},  B = 1 + 0.1 * (n - 1),  HI = 1 - T_P  (Eq. 9)
report : Q = mean stated confidence,  U = max_l (1 - A_l)            (Eq. 10)

The scorer works on any table with one row per image and the columns

    Plant_Stem_Number, Plant Images Level, File Name,
    Health Status, Disease Category, Disease, Severity, Confidence

It accepts both the annotation workbook (plain column names) and the VLM
results workbook (``<column> (GT)`` / ``<column> (Predicted)``), selected with
``source``. Scores are computed per plant-session; the session is the date
prefix of the image file name (``Date_03_Feb_2026_...``).

All coefficients were set by three agronomists (Section 4.3.6). Disease-specific
threat coefficients follow Supplementary Tables A8-A12.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# =============================================================================
# 1) COEFFICIENTS
# =============================================================================

# Severity S_i (Supplementary Table A13).
SEVERITY_SCORE = {
    "n/a": 0.00,
    "na": 0.00,
    "": 0.00,
    "uncertain": 0.00,
    "low": 0.30,
    "medium": 0.60,
    "high": 0.85,
}

# Stated confidence C_i (Supplementary Table A14). Reported through Q only;
# it is not multiplied into the threat.
CONFIDENCE_SCORE = {
    "": 1.00,          # healthy rows without a stated confidence
    "low": 0.65,
    "medium": 0.85,
    "high": 1.00,
}

# Vertical canopy weight V_l.
VERTICAL_WEIGHT = {
    "high": 1.00,
    "top": 1.00,
    "middle": 0.85,
    "mid": 0.85,
    "low": 0.65,
    "bottom": 0.65,
}

# Spread multiplier m_i.
# Nematode conditions are scored as "other" (also evaluated: dedicated
# nematode class with m = 1.15).
SPREAD_MULTIPLIER = {
    "healthy": 1.00,
    "nutrient": 1.00,
    "abiotic": 1.00,
    "uncertain": 1.00,
    "other": 1.00,
    "fungal": 1.35,
    "bacterial": 1.50,
    "viral": 1.70,
}

# Category baseline of E_i, used when the diagnosis is not listed below.
# (Also evaluated: dedicated nematode class with E = 0.60.)
CATEGORY_E_SCORE = {
    "healthy": 0.00,
    "abiotic": 0.45,
    "nutrient": 0.40,
    "fungal": 0.80,
    "bacterial": 0.90,
    "viral": 0.95,
    "uncertain": 0.45,
    "other": 0.65,
}

# Disease-specific threat coefficient E_i, matched as a keyword in the
# diagnosis text. First match wins, so specific names precede generic ones.
DISEASE_E_OVERRIDES: List[Tuple[str, float]] = [
    # --- Viral (Supplementary Table A8) ---
    ("yellow leaf curl", 1.00),
    ("tylcv", 1.00),
    ("spotted wilt", 0.95),
    ("tswv", 0.95),
    ("cucumber mosaic", 0.85),       # also evaluated: 0.90
    ("tobacco mosaic", 0.90),
    ("tomato mosaic", 0.90),
    ("potato virus y", 0.85),
    ("bushy stunt", 0.80),
    ("viroid", 0.90),
    ("tobamovirus", 0.90),
    ("mosaic", 0.90),
    ("mottle", 0.90),
    ("viral", 0.95),
    # --- Bacterial (Supplementary Table A9) ---
    ("bacterial wilt", 0.95),
    ("bacterial canker", 0.95),
    ("bacterial spot", 0.90),
    ("bacterial speck", 0.85),
    ("stem & fruit rot", 0.90),
    ("syringae", 0.80),
    ("pith necrosis", 0.75),          # also evaluated: 0.90 (category baseline)
    ("big bud", 0.90),
    ("phytoplasma", 0.90),
    ("aster yellows", 0.85),
    # --- Fungal (Supplementary Table A10) ---
    ("late blight", 1.00),
    ("early blight", 0.80),
    ("septoria", 0.75),
    ("leaf mold", 0.75),
    ("powdery mildew", 0.70),
    ("gray mold", 0.85),
    ("anthracnose", 0.75),            # also evaluated: 0.80 (category baseline)
    ("target spot", 0.75),            # also evaluated: 0.80 (category baseline)
    ("cercospora", 0.70),
    ("fusarium", 0.85),
    ("verticillium", 0.80),
    ("southern blight", 0.85),
    ("buckeye rot", 0.75),
    ("white mold", 0.80),
    ("seedling blight", 0.75),
    # --- Nutrient (Supplementary Table A12) ---
    ("potassium", 0.45),
    ("magnesium", 0.35),
    ("calcium", 0.50),
    ("nitrogen", 0.40),
    ("phosphorus", 0.35),             # also evaluated: 0.40
    ("iron deficiency", 0.30),        # also evaluated: 0.40 (category baseline)
    ("zinc", 0.30),
    ("boron", 0.40),
    ("chlorosis", 0.35),
    # --- Abiotic / physiological (Supplementary Table A11) ---
    ("physiological leaf roll", 0.30),
    ("physiological leaf curl", 0.30),
    ("leaf roll", 0.30),
    ("leaf curl", 0.30),
    ("edema", 0.30),
    ("oedema", 0.30),
    ("mechanical", 0.35),
    ("physical", 0.35),
    ("surface injury", 0.35),
    ("water-logging", 0.55),
    ("waterlogging", 0.55),
    ("drought", 0.55),                # also evaluated: 0.40
    ("herbicide", 0.60),              # also evaluated: 0.45
    ("ozone", 0.45),
    ("air pollution", 0.45),
    ("sunscald", 0.45),
    ("heat stress", 0.45),
    ("tip burn", 0.45),
    ("edge burn", 0.45),
    ("marginal scorch", 0.45),
    ("leaf scorch", 0.45),
    ("leaf miner", 0.45),
]

# Level-fusion weights W_l and worst-level weight beta (Eq. 8).
PLANT_LEVEL_WEIGHT = {
    "high": 0.50,
    "middle": 0.30,
    "low": 0.20,
}
BETA = 0.75

# Worst-leaf blend weight alpha_l (Eq. 6).
ALPHA_INFECTIOUS = 0.75
ALPHA_OTHER = 0.30

# Alert bands on the plant threat T_P (Section 4.3.4).
ALERT_THRESHOLDS = [
    (0.85, "Critical"),
    (0.55, "High"),
    (0.30, "Medium"),
    (0.10, "Low"),
    (0.00, "Minimal"),
]

FIELDS = ["Health Status", "Disease Category", "Disease", "Severity", "Confidence"]
IMAGE_NAME_RE = re.compile(r"\.(?:jpe?g|png|bmp|tiff?|webp)$", re.IGNORECASE)
SESSION_RE = re.compile(r"(Date_\d{2}_[A-Za-z]{3}_\d{4})")


# =============================================================================
# 2) HELPERS
# =============================================================================
def norm(x) -> str:
    """Normalise a cell value to a lowercase, stripped string."""
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def canonical_level(level: str) -> str:
    x = norm(level)
    if x in {"high", "top"}:
        return "high"
    if x in {"middle", "mid"}:
        return "middle"
    if x in {"low", "bottom"}:
        return "low"
    return x or "unknown"


def disease_group(category: str, disease: str, health_status: str) -> str:
    """Map category and diagnosis text to a biological group."""
    cat = norm(category)
    dis = norm(disease)
    health = norm(health_status)

    if health == "healthy":
        return "healthy"
    if "viral" in cat or "virus" in dis or "tylcv" in dis or "mosaic" in dis:
        return "viral"
    if "fungal" in cat or "blight" in dis or "septoria" in dis or "mildew" in dis or "mold" in dis:
        return "fungal"
    if "bacterial" in cat or "bacterial" in dis:
        return "bacterial"
    if "nutrient" in cat or "deficiency" in dis or "chlorosis" in dis:
        return "nutrient"
    if "abiotic" in cat or any(k in dis for k in ["physical", "mechanical", "roll", "curl", "scorch",
                                                  "burn", "edema", "oedema", "sunscald", "leaf miner"]):
        return "abiotic"
    if "uncertain" in cat or "uncertain" in dis:
        return "uncertain"
    return "other"


def exact_disease_score(category: str, disease: str, health_status: str) -> float:
    """E_i: disease-specific coefficient, else the category baseline."""
    health = norm(health_status)
    dis = norm(disease)
    if health == "healthy":
        return 0.0
    for key, score in DISEASE_E_OVERRIDES:
        if key in dis:
            return float(score)
    group = disease_group(category, disease, health_status)
    return float(CATEGORY_E_SCORE.get(group, CATEGORY_E_SCORE["other"]))


def severity_score(severity: str) -> float:
    return float(SEVERITY_SCORE.get(norm(severity), 0.0))


def confidence_score(confidence: str, health_status: str) -> float:
    x = norm(confidence)
    if x == "" and norm(health_status) == "healthy":
        return 1.0
    return float(CONFIDENCE_SCORE.get(x, 0.85))


def spread_multiplier(group: str) -> float:
    return float(SPREAD_MULTIPLIER.get(group, 1.00))


def alert_level(score: float) -> str:
    for threshold, label in ALERT_THRESHOLDS:
        if score >= threshold:
            return label
    return "Minimal"


def disease_agreement(labels: List[str]) -> float:
    """Diagnosis agreement D_l over the N_l = 3 images of a canopy level.

    1.00  no diseased image, or all images share one diagnosis
    0.75  one diagnosis is held by a majority
    0.50  all images diseased, no majority
    0.25  otherwise

    A proportion-based variant (share of the modal diagnosis) was also
    evaluated; the banded rule above is the one used for all reported results.
    """
    labels = [x for x in labels if x and x != "healthy"]
    if len(labels) == 0:
        return 1.00
    counts = pd.Series(labels).value_counts()
    top_count = int(counts.iloc[0])
    unique_count = len(counts)
    if top_count == 3:
        return 1.00
    if top_count == 2:
        return 0.75
    if unique_count == 3:
        return 0.50
    return 0.25


def severity_agreement(severities: List[str]) -> float:
    """Severity agreement K_l over images with S_i > 0.

    1.00 if at most one such image or all severities equal,
    0.75 if they span two adjacent bands, 0.40 if Low and High are both present.
    """
    vals = [severity_score(x) for x in severities if severity_score(x) > 0]
    if len(vals) <= 1:
        return 1.00
    spread = max(vals) - min(vals)
    if spread == 0:
        return 1.00
    if spread <= 0.30:
        return 0.75
    return 0.40


def dominant_issue(group: pd.DataFrame) -> str:
    """Issue k_l of the diseased image with the largest R_i * m_i."""
    diseased = group[group["Health Status"].fillna("").astype(str).str.lower().str.strip() != "healthy"]
    if diseased.empty:
        return "Healthy"
    idx = (diseased["Image_Risk"] * diseased["Spread_Multiplier"]).idxmax()
    disease = str(diseased.loc[idx, "Disease"] or "Uncertain")
    category = str(diseased.loc[idx, "Disease Category"] or "Uncertain")
    return f"{category}: {disease}"


def session_from_filename(name: str, default: str = "") -> str:
    """Return the session tag (``Date_03_Feb_2026``) encoded in an image file name."""
    m = SESSION_RE.search(str(name))
    return m.group(1) if m else default


# =============================================================================
# 3) LOADING
# =============================================================================
def load_week(path: Path, sheet_name: str, source: str = "annotation",
              header_row: int = 1) -> pd.DataFrame:
    """Load one sheet as one row per image with standard column names.

    source:
        "annotation"  plain columns (``Health Status``), e.g. the dataset workbook
        "gt"          ``Health Status (GT)`` columns of the VLM results workbook
        "predicted"   ``Health Status (Predicted)`` columns of the VLM results workbook
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, engine="openpyxl")
    return prepare_frame(df, source=source, default_session=sheet_name)


def prepare_frame(df: pd.DataFrame, source: str = "annotation",
                  default_session: str = "") -> pd.DataFrame:
    suffix = {"annotation": "", "gt": " (GT)", "predicted": " (Predicted)"}[source]
    df = df.dropna(how="all").copy()

    # Plant IDs are written on the first row of each plant block only.
    df["Plant_Stem_Number"] = df["Plant_Stem_Number"].ffill()

    # Keep only rows that reference an image file.
    df = df[df["File Name"].astype(str).str.strip().str.contains(IMAGE_NAME_RE)].copy()

    if suffix:
        df = df.rename(columns={f"{c}{suffix}": c for c in FIELDS})
    for c in FIELDS:
        if c not in df.columns:
            df[c] = pd.NA

    df["Session"] = df["File Name"].apply(lambda f: session_from_filename(f, default_session))
    df["Level"] = df["Plant Images Level"].apply(canonical_level)
    return df


# =============================================================================
# 4) SCORING ENGINE
# =============================================================================
def compute_image_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Disease_Group"] = df.apply(
        lambda r: disease_group(r.get("Disease Category"), r.get("Disease"), r.get("Health Status")), axis=1)
    df["E"] = df.apply(
        lambda r: exact_disease_score(r.get("Disease Category"), r.get("Disease"), r.get("Health Status")), axis=1)
    df["S"] = df["Severity"].apply(severity_score)
    df["C"] = df.apply(lambda r: confidence_score(r.get("Confidence"), r.get("Health Status")), axis=1)
    df["Image_Risk"] = df["E"] * df["S"]
    df["Reliability"] = df["C"]
    df["Spread_Multiplier"] = df["Disease_Group"].apply(spread_multiplier)
    df["Is_Infectious"] = df["Disease_Group"].isin(["viral", "fungal", "bacterial"])
    df["Review_Uncertain_Severity"] = df["Severity"].apply(norm) == "uncertain"
    return df


def compute_level_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (session, plant, level), g in df.groupby(["Session", "Plant_Stem_Number", "Level"], dropna=False):
        v = VERTICAL_WEIGHT.get(level, 1.0)
        r_values = g["Image_Risk"].fillna(0).astype(float)
        r_max = float(r_values.max())
        r_mean = float(r_values.mean())

        d_agree = disease_agreement(g["Disease"].fillna("Healthy").astype(str).str.lower().tolist())
        s_agree = severity_agreement(g["Severity"].fillna("").astype(str).tolist())
        agreement = 0.7 * d_agree + 0.3 * s_agree
        g_mult = 0.8 + 0.3 * agreement

        infectious_present = bool(g["Is_Infectious"].any())
        alpha = ALPHA_INFECTIOUS if infectious_present else ALPHA_OTHER
        m_level = float(g["Spread_Multiplier"].max())
        core = alpha * r_max + (1 - alpha) * r_mean
        level_threat = min(v * g_mult * m_level * core, 1.0)

        rows.append({
            "Session": session,
            "Plant_Stem_Number": plant,
            "Level": level,
            "Image_Count": len(g),
            "Vertical_Weight_V": v,
            "R_Max": r_max,
            "R_Mean": r_mean,
            "Disease_Agreement_D": d_agree,
            "Severity_Agreement_K": s_agree,
            "Agreement_A": agreement,
            "Agreement_Multiplier_G": g_mult,
            "Infectious_Present": infectious_present,
            "Alpha": alpha,
            "Spread_Multiplier_M": m_level,
            "Level_Threat_T": level_threat,
            "Level_Reliability_Q": float(g["Reliability"].mean()),
            "Level_Uncertainty_U": 1 - agreement,
            "Dominant_Issue": dominant_issue(g),
        })
    return pd.DataFrame(rows)


def systemic_boost(level_scores: pd.DataFrame) -> Tuple[float, str, int]:
    """Systemic boost B = 1 + 0.1 (n - 1).

    The issue is the specific nutrient for nutrient deficiencies (magnesium,
    potassium, other nutrient) and the category for all other conditions.
    """
    candidates = []
    for _, r in level_scores.iterrows():
        issue = str(r["Dominant_Issue"]).lower()
        if r["Level_Threat_T"] <= 0:
            continue
        if "viral" in issue:
            candidates.append("viral")
        elif "fungal" in issue or "septoria" in issue:
            candidates.append("fungal")
        elif "bacterial" in issue:
            candidates.append("bacterial")
        elif "magnesium" in issue:
            candidates.append("magnesium")
        elif "potassium" in issue:
            candidates.append("potassium")
        elif "nutrient" in issue:
            candidates.append("nutrient")
        elif "abiotic" in issue:
            candidates.append("abiotic")
        else:
            candidates.append("other")

    if not candidates:
        return 1.0, "None", 0
    counts = pd.Series(candidates).value_counts()
    dominant = str(counts.index[0])
    n_levels = int(counts.iloc[0])
    boost = 1 + 0.1 * max(n_levels - 1, 0)
    return float(boost), dominant, n_levels


def compute_plant_scores(level_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (session, plant), g in level_df.groupby(["Session", "Plant_Stem_Number"], dropna=False):
        score_by_level = {row["Level"]: row["Level_Threat_T"] for _, row in g.iterrows()}
        t_high = float(score_by_level.get("high", 0.0))
        t_mid = float(score_by_level.get("middle", 0.0))
        t_low = float(score_by_level.get("low", 0.0))

        max_t = max(t_high, t_mid, t_low)
        weighted_structural = (
            PLANT_LEVEL_WEIGHT["high"] * t_high
            + PLANT_LEVEL_WEIGHT["middle"] * t_mid
            + PLANT_LEVEL_WEIGHT["low"] * t_low
        )
        p_base = BETA * max_t + (1 - BETA) * weighted_structural
        boost, boosted_issue, boosted_levels = systemic_boost(g)
        p_final = min(p_base * boost, 1.0)

        # Infectious overrides change the alert level only, not the HI.
        viral_present = any(g["Dominant_Issue"].astype(str).str.lower()
                            .str.contains("viral|virus|tylcv|mosaic", regex=True))
        fungal_top = False
        high_row = g[g["Level"] == "high"]
        if not high_row.empty:
            fungal_top = bool(high_row["Dominant_Issue"].astype(str).str.lower()
                              .str.contains("fungal|septoria|blight|mildew|mold", regex=True).any())

        alert = alert_level(p_final)
        review_flag = ""
        if viral_present and alert in {"Minimal", "Low"}:
            alert = "Medium"
            review_flag = "Viral review override"
        if fungal_top and alert in {"Minimal", "Low", "Medium"}:
            alert = "High"
            review_flag = "Top fungal override"

        rows.append({
            "Session": session,
            "Plant_Stem_Number": plant,
            "T_High": t_high,
            "T_Middle": t_mid,
            "T_Low": t_low,
            "Max_Level_Threat": max_t,
            "Weighted_Structural_Threat": weighted_structural,
            "P_Base": p_base,
            "Systemic_Boost_B": boost,
            "Boosted_Issue": boosted_issue,
            "Boosted_Levels_n": boosted_levels,
            "P_Final": p_final,
            "Health_Index": 1.0 - p_final,
            "Alert_Level": alert,
            "Review_Flag": review_flag,
            "Plant_Reliability_Q": float(g["Level_Reliability_Q"].mean()),
            "Plant_Uncertainty_U": float(g["Level_Uncertainty_U"].max()),
            "Main_Issue": str(g.sort_values("Level_Threat_T", ascending=False).iloc[0]["Dominant_Issue"]),
        })
    return pd.DataFrame(rows)


def score_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score a prepared image table. Returns (image, level, plant) tables."""
    image_scores = compute_image_scores(df)
    level_scores = compute_level_scores(image_scores)
    plant_scores = compute_plant_scores(level_scores)
    return image_scores, level_scores, plant_scores


def scoring_config_table() -> pd.DataFrame:
    """All coefficients as one long table (written to the output workbook)."""
    rows = []
    for name, table in [("S", SEVERITY_SCORE), ("C", CONFIDENCE_SCORE), ("V", VERTICAL_WEIGHT),
                        ("m", SPREAD_MULTIPLIER), ("E_category", CATEGORY_E_SCORE),
                        ("W", PLANT_LEVEL_WEIGHT)]:
        rows += [{"Parameter": name, "Key": k, "Value": v} for k, v in table.items()]
    rows += [{"Parameter": "E_disease", "Key": k, "Value": v} for k, v in DISEASE_E_OVERRIDES]
    rows += [{"Parameter": "beta", "Key": "", "Value": BETA},
             {"Parameter": "alpha", "Key": "infectious", "Value": ALPHA_INFECTIOUS},
             {"Parameter": "alpha", "Key": "other", "Value": ALPHA_OTHER}]
    rows += [{"Parameter": "alert", "Key": label, "Value": t} for t, label in ALERT_THRESHOLDS]
    return pd.DataFrame(rows)


def write_scored_workbook(output_file: Path, image_scores: pd.DataFrame,
                          level_scores: pd.DataFrame, plant_scores: pd.DataFrame) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        plant_scores.sort_values(["Session", "Plant_Stem_Number"]).to_excel(
            writer, sheet_name="Plant_Summary", index=False)
        level_scores.sort_values(["Session", "Plant_Stem_Number", "Level"]).to_excel(
            writer, sheet_name="Level_Scores", index=False)
        image_scores.sort_values(["Session", "Plant_Stem_Number", "Level", "File Name"]).to_excel(
            writer, sheet_name="Image_Scores", index=False)
        scoring_config_table().to_excel(writer, sheet_name="Scoring_Config", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 42)


def score_workbook(input_file: Path, sheet_names: Sequence[str], source: str = "annotation",
                   output_file: Optional[Path] = None, sessions: Optional[Iterable[str]] = None,
                   header_row: int = 1):
    """Score one or more sheets of a workbook and optionally write the result.

    ``sessions`` keeps only images whose file name carries one of the given
    session tags (e.g. the four scouting dates).
    """
    frames = [load_week(Path(input_file), s, source=source, header_row=header_row) for s in sheet_names]
    df = pd.concat(frames, ignore_index=True)
    if sessions:
        df = df[df["Session"].isin(list(sessions))]
    image_scores, level_scores, plant_scores = score_frame(df)
    if output_file is not None:
        write_scored_workbook(Path(output_file), image_scores, level_scores, plant_scores)
    return image_scores, level_scores, plant_scores
