"""
Results workbook: agronomist annotations next to each model's predictions.

Layout of every model sheet (row 1 title, row 2 header, one row per image):

    Plant_Stem_Number | Plant Images Level | File Name |
    Health Status (GT) | Health Status (Predicted) | ... | Reasoning (GT) | Reasoning (Predicted)

This is the input of ``scripts/05_evaluate_vlms.py`` and, with
``--source predicted``, of ``scripts/06_score_health_index.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from .vlm import load_jsonl_results

FIELDS = ["Health Status", "Disease Category", "Disease", "Severity",
          "Confidence", "Visible Symptoms", "Reasoning"]

# JSON key returned by the VLM for each workbook field (configs/prompts/user_prompt.txt).
JSON_KEYS = {
    "Health Status": "plant_health_status",
    "Disease Category": "disease_category",
    "Disease": "disease",
    "Severity": "severity",
    "Confidence": "confidence",
    "Visible Symptoms": "visible_symptoms",
    "Reasoning": "reasoning",
}


def load_annotations(path: Path, sheets: Optional[Sequence[str]] = None, header_row: int = 1) -> pd.DataFrame:
    """Stack the session sheets of the annotation workbook into one image table."""
    xl = pd.ExcelFile(path)
    sheets = list(sheets) if sheets else [s for s in xl.sheet_names if s.lower().startswith("week")]
    frames = []
    for sheet in sheets:
        df = pd.read_excel(path, sheet_name=sheet, header=header_row, dtype=object)
        df = df.dropna(how="all").copy()
        df["Plant_Stem_Number"] = df["Plant_Stem_Number"].ffill()
        df = df[df["File Name"].astype(str).str.contains(r"\.(?:jpe?g|png|bmp|tiff?|webp)$", case=False)]
        frames.append(df[["Plant_Stem_Number", "Plant Images Level", "File Name"] + FIELDS])
    return pd.concat(frames, ignore_index=True)


def _value(result: dict, field: str):
    v = result.get(JSON_KEYS[field], "")
    if isinstance(v, list):
        return "; ".join(str(x) for x in v)
    return v


def merge_predictions(annotations: pd.DataFrame, results: Dict[str, dict]) -> pd.DataFrame:
    """Interleave GT and predicted columns for one model."""
    out = annotations[["Plant_Stem_Number", "Plant Images Level", "File Name"]].copy()
    for field in FIELDS:
        out[f"{field} (GT)"] = annotations[field].to_numpy()
        out[f"{field} (Predicted)"] = [
            _value(results[f], field) if f in results else None for f in annotations["File Name"]
        ]
    return out


def write_results_workbook(annotations: pd.DataFrame, jsonl_dir: Path, models: Sequence[str],
                           output: Path, title: str = "AI-VS3 VLM benchmark") -> Dict[str, int]:
    """Write one sheet per model; returns the number of answered images per model."""
    answered = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for model in models:
            path = Path(jsonl_dir) / f"{model}.jsonl"
            results = load_jsonl_results(path) if path.exists() else {}
            sheet = merge_predictions(annotations, results)
            answered[model] = int(sheet["Health Status (Predicted)"].notna().sum())
            sheet.to_excel(writer, sheet_name=model[:31], index=False, startrow=1)
            ws = writer.sheets[model[:31]]
            ws.cell(row=1, column=1, value=f"{title} - {model}")
            ws.freeze_panes = "D3"
    return answered
