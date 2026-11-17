"""
VLM benchmark evaluation (manuscript Section 5.2, Table 4).

Each model sheet of the results workbook holds, per image, the agronomist
annotation (``<field> (GT)``) next to the model output (``<field> (Predicted)``).

Categorical fields are compared after label normalisation:

* Health Status     Healthy / Not Healthy
* Disease Category  biological group; a healthy leaf counts as "no disease"
* Disease           canonical diagnosis; a healthy leaf counts as "none"
* Severity          Low / Medium / High; a healthy leaf counts as "none"
* Confidence        Low / Medium / High

A missing or unparsable prediction is counted as incorrect. Free-text fields
(Visible Symptoms, Reasoning) are compared with BLEU-4, ROUGE-L F1 and TF-IDF
cosine similarity.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, f1_score

INVALID_STRINGS = {"", "-", "--", "na", "n/a", "none", "null", "nan", "<na>"}
IMAGE_NAME_RE = re.compile(r"\.(?:jpe?g|png|bmp|tiff?|webp)$", re.IGNORECASE)

ORDINAL_MAP = {"low": 1, "medium": 2, "high": 3}

# Canonical diagnosis labels. First match wins.
DISEASE_ALIAS_PATTERNS: List[Tuple[str, str]] = [
    (r"\bmagnesium\b|\bmg deficiency\b", "magnesium deficiency"),
    (r"\bpotassium\b|\bk deficiency\b", "potassium deficiency"),
    (r"\bnitrogen\b|\bn deficiency\b", "nitrogen deficiency"),
    (r"\biron\b|\bfe deficiency\b", "iron deficiency"),
    (r"\bphosphorus\b|\bp deficiency\b", "phosphorus deficiency"),
    (r"\bcalcium\b|\bca deficiency\b", "calcium deficiency"),
    (r"\bearly blight\b", "early blight"),
    (r"\bseptoria\b", "septoria leaf spot"),
    (r"\btarget spot\b", "target spot"),
    (r"\bpowdery mildew\b", "powdery mildew"),
    (r"\bbacterial spot\b", "bacterial spot"),
    (r"\bbacterial speck\b", "bacterial speck"),
    (r"\btomato yellow leaf curl\b|\btylcv\b", "tomato yellow leaf curl virus"),
    (r"\btomato mosaic\b|\btobamovirus\b|\bmosaic virus\b", "tomato mosaic virus"),
    (r"\bphysiological leaf roll\b|\bleaf roll\b", "physiological leaf roll"),
    (r"\bphysiological leaf curl\b", "physiological leaf curl"),
    (r"\bleaf curl\b", "physiological leaf curl"),
    (r"\bleaf miner\b", "leaf miner damage"),
    (r"\bsunscald\b", "sunscald"),
    (r"\bleaf scorch\b|\bmarginal scorch\b|\btip burn\b|\bmarginal burn\b", "leaf scorch / tip burn"),
    (r"\bedema\b|\boedema\b", "edema"),
    (r"\bdrought\b", "drought stress"),
    (r"\benvironmental stress\b|\babiotic stress\b", "environmental stress"),
    (r"\bphysical\b|\bmechanical\b|\binsect feeding\b", "physical / mechanical damage"),
    (r"\buncertain\b|\bunknown\b", "uncertain"),
    (r"\bnutrient deficiency\b|\bnutrient disorder\b", "nutrient deficiency"),
]

CATEGORICAL_FIELDS = ["Health Status", "Disease Category", "Disease", "Severity", "Confidence"]
TEXT_FIELDS = ["Visible Symptoms", "Reasoning"]


# =============================================================================
# Label normalisation
# =============================================================================
def clean_text_cell(value) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return None if s.lower() in INVALID_STRINGS else s


def label_base(value) -> Optional[str]:
    s = clean_text_cell(value)
    if s is None:
        return None
    s = s.lower().replace("_", " ").replace("/", " / ")
    s = re.sub(r"[^a-z0-9/+ -]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return None if s in INVALID_STRINGS else s


def normalize_health(value) -> Optional[str]:
    s = label_base(value)
    if s is None:
        return None
    if "not healthy" in s or "unhealthy" in s or "diseased" in s or "disease" in s:
        return "not healthy"
    if "healthy" in s:
        return "healthy"
    return s


def normalize_ordinal(value) -> Optional[str]:
    s = label_base(value)
    if s is None:
        return None
    if "low" in s:
        return "low"
    if "med" in s or "moderate" in s:
        return "medium"
    if "high" in s or "severe" in s:
        return "high"
    return s


def normalize_disease_category(value) -> Optional[str]:
    s = label_base(value)
    if s is None:
        return None
    if "nutrient" in s or "deficiency" in s:
        return "nutrient disease"
    if "fung" in s or "mildew" in s or "blight" in s or "septoria" in s or "target spot" in s:
        return "fungal leaf disease"
    if "viral" in s or "virus" in s or "mosaic" in s or "tylcv" in s:
        return "viral leaf disease"
    if "bacter" in s:
        return "bacterial leaf disease"
    if "abiotic" in s or "environment" in s or "physiological" in s or "sunscald" in s or "scorch" in s:
        return "abiotic"
    if "nematode" in s:
        return "nematode"
    if "uncertain" in s or "unknown" in s:
        return "uncertain"
    return s


def normalize_disease_name(value) -> Optional[str]:
    s = label_base(value)
    if s is None:
        return None
    for pattern, canonical in DISEASE_ALIAS_PATTERNS:
        if re.search(pattern, s):
            return canonical
    s = re.sub(r"\b(symptoms?|disease|like|mild|moderate|severe)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# =============================================================================
# Text similarity
# =============================================================================
def tokenize(text: Optional[str]) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower()) if text is not None else []


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1)))


def sentence_bleu4(reference: Optional[str], hypothesis: Optional[str]) -> float:
    """Sentence BLEU-4 with add-one smoothing."""
    ref, hyp = tokenize(reference), tokenize(hypothesis)
    if not ref:
        return np.nan
    if not hyp:
        return 0.0
    precisions = []
    for n in range(1, 5):
        hyp_ng, ref_ng = _ngrams(hyp, n), _ngrams(ref, n)
        total = sum(hyp_ng.values())
        if total == 0:
            precisions.append(1.0 / (len(hyp) + 1.0))
            continue
        clipped = sum(min(c, ref_ng.get(g, 0)) for g, c in hyp_ng.items())
        precisions.append((clipped + 1.0) / (total + 1.0))
    log_p = sum(math.log(max(p, 1e-12)) for p in precisions) / 4.0
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1.0 - len(ref) / max(len(hyp), 1))
    return float(bp * math.exp(log_p))


def rouge_l_f1(reference: Optional[str], hypothesis: Optional[str]) -> float:
    ref, hyp = tokenize(reference), tokenize(hypothesis)
    if not ref:
        return np.nan
    if not hyp:
        return 0.0
    m, n = len(ref), len(hyp)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if ref[i - 1] == hyp[j - 1] else max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[n]
    if lcs == 0:
        return 0.0
    p, r = lcs / n, lcs / m
    return float(2 * p * r / (p + r))


def rowwise_tfidf_cosine(gt_texts: List[str], pred_texts: List[str]) -> np.ndarray:
    """Cosine between paired texts in one shared TF-IDF space (unigrams + bigrams)."""
    n = len(gt_texts)
    if n == 0:
        return np.array([], dtype=float)
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2))
    try:
        mat = vec.fit_transform([t or "" for t in gt_texts + pred_texts])
    except ValueError:
        return np.zeros(n, dtype=float)
    return np.asarray(mat[:n].multiply(mat[n:]).sum(axis=1)).ravel().astype(float)


# =============================================================================
# Workbook loading
# =============================================================================
def read_results_workbook(path: Path, header_row: int = 1,
                          exclude_sheets: Sequence[str] = ("Master_Template",)) -> Dict[str, pd.DataFrame]:
    """Return {model sheet name: image rows} from the results workbook."""
    xl = pd.ExcelFile(path)
    excluded = {s.lower() for s in exclude_sheets}
    out = {}
    for sheet in xl.sheet_names:
        if sheet.lower() in excluded:
            continue
        df = pd.read_excel(path, sheet_name=sheet, header=header_row, dtype=object)
        if "Health Status (GT)" not in df.columns:
            continue
        df = df[df["File Name"].astype(str).str.strip().str.contains(IMAGE_NAME_RE)]
        df = df[df["Health Status (GT)"].apply(normalize_health).notna()]
        out[sheet] = df.reset_index(drop=True)
    return out


def normalise_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalised GT / prediction labels for every categorical field."""
    out = pd.DataFrame({"File Name": df["File Name"]})
    for side, tag in (("gt", "GT"), ("pred", "Predicted")):
        health = df[f"Health Status ({tag})"].apply(normalize_health)
        out[f"health_{side}"] = health

        def with_none(col: str, fn, fill: str) -> pd.Series:
            v = df[f"{col} ({tag})"].apply(fn)
            return v.where(v.notna(), np.where(health == "healthy", fill, None))

        out[f"category_{side}"] = with_none("Disease Category", normalize_disease_category, "no disease")
        out[f"disease_{side}"] = with_none("Disease", normalize_disease_name, "none")
        out[f"severity_{side}"] = with_none("Severity", normalize_ordinal, "none")
        out[f"confidence_{side}"] = df[f"Confidence ({tag})"].apply(normalize_ordinal)
    return out


# =============================================================================
# Metrics
# =============================================================================
def categorical_scores(gt: pd.Series, pred: pd.Series) -> Dict[str, float]:
    mask = gt.notna()
    y_true = gt[mask].astype(str)
    y_pred = pred[mask].fillna("<missing>").astype(str)
    labels = sorted(y_true.unique())
    return {
        "n": int(mask.sum()),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else np.nan,
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)) if y_true.nunique() > 1 else np.nan,
    }


def text_scores(gt_texts: pd.Series, pred_texts: pd.Series) -> Dict[str, float]:
    gt = gt_texts.apply(clean_text_cell)
    mask = gt.notna()
    g = gt[mask].astype(str).tolist()
    p = pred_texts[mask].apply(clean_text_cell).fillna("").astype(str).tolist()
    return {
        "bleu4": float(np.nanmean([sentence_bleu4(a, b) for a, b in zip(g, p)])) if g else np.nan,
        "rouge_l": float(np.nanmean([rouge_l_f1(a, b) for a, b in zip(g, p)])) if g else np.nan,
        "cosine": float(np.mean(rowwise_tfidf_cosine(g, p))) if g else np.nan,
    }


def evaluate_model(df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Return (summary row, per-image normalised table) for one model sheet."""
    norm = normalise_sheet(df)
    row: Dict[str, float] = {"images": len(df)}
    for key in ("health", "category", "disease", "severity", "confidence"):
        s = categorical_scores(norm[f"{key}_gt"], norm[f"{key}_pred"])
        row[f"{key}_accuracy"] = 100 * s["accuracy"]
        row[f"{key}_f1_macro"] = s["f1_macro"]
        row[f"{key}_kappa"] = s["kappa"]
        norm[f"{key}_correct"] = (norm[f"{key}_gt"] == norm[f"{key}_pred"]).where(norm[f"{key}_gt"].notna())
    for field in TEXT_FIELDS:
        tag = field.lower().replace(" ", "_")
        for metric, value in text_scores(df[f"{field} (GT)"], df[f"{field} (Predicted)"]).items():
            row[f"{tag}_{metric}"] = value
    return row, norm


def confusion_table(gt: pd.Series, pred: pd.Series) -> pd.DataFrame:
    mask = gt.notna()
    return pd.crosstab(gt[mask].rename("Ground truth"),
                       pred[mask].fillna("<missing>").rename("Prediction"))


def evaluate_workbook(path: Path, header_row: int = 1) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Evaluate every model sheet. Returns (summary table, per-model image tables)."""
    rows, per_image = [], {}
    for model, df in read_results_workbook(path, header_row=header_row).items():
        summary, norm = evaluate_model(df)
        rows.append({"model": model, **summary})
        per_image[model] = norm
    return pd.DataFrame(rows), per_image


TABLE4_COLUMNS = {
    "health_accuracy": "Health Status",
    "category_accuracy": "Disease Category",
    "disease_accuracy": "Disease",
    "severity_accuracy": "Severity",
    "confidence_accuracy": "Confidence",
    "reasoning_bleu4": "BLEU-4",
    "reasoning_rouge_l": "ROUGE-L",
    "reasoning_cosine": "Cosine",
}


def table4(summary: pd.DataFrame, text_field: str = "reasoning") -> pd.DataFrame:
    """Format the summary in the layout of manuscript Table 4."""
    cols = {k.replace("reasoning", text_field): v for k, v in TABLE4_COLUMNS.items()}
    out = summary[["model"] + list(cols)].rename(columns={"model": "VLM", **cols})
    return out.round(4).reset_index(drop=True)
