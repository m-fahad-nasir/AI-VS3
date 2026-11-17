import pandas as pd
import pytest

from aivs3 import evaluation as ev


@pytest.mark.parametrize("value, expected", [
    ("Healthy", "healthy"), ("Not Healthy", "not healthy"), ("-", None), (None, None)])
def test_normalize_health(value, expected):
    assert ev.normalize_health(value) == expected


@pytest.mark.parametrize("value, expected", [
    ("Nutrient", "nutrient disease"), ("Nutrient Disease", "nutrient disease"),
    ("Fungal", "fungal leaf disease"), ("Viral Leaf Disease", "viral leaf disease"),
    ("Abiotic", "abiotic")])
def test_normalize_category(value, expected):
    assert ev.normalize_disease_category(value) == expected


@pytest.mark.parametrize("value, expected", [
    ("Mild magnesium deficiency", "magnesium deficiency"),
    ("Tomato yellow leaf curl virus (TYLCV)", "tomato yellow leaf curl virus"),
    ("Leaf tip burn / marginal scorch", "leaf scorch / tip burn")])
def test_normalize_disease_name(value, expected):
    assert ev.normalize_disease_name(value) == expected


def test_text_metrics_identical_and_disjoint():
    text = "interveinal chlorosis on older leaves"
    assert ev.sentence_bleu4(text, text) == pytest.approx(1.0)
    assert ev.rouge_l_f1(text, text) == pytest.approx(1.0)
    assert ev.rouge_l_f1(text, "healthy green leaf") == 0.0
    assert ev.rowwise_tfidf_cosine([text], [text])[0] == pytest.approx(1.0)


def test_evaluate_model_counts_healthy_as_no_disease():
    def row(h_gt, c_gt, h_p, c_p, s_gt=None, s_p=None):
        return {"File Name": "x.jpg",
                "Health Status (GT)": h_gt, "Health Status (Predicted)": h_p,
                "Disease Category (GT)": c_gt, "Disease Category (Predicted)": c_p,
                "Disease (GT)": None, "Disease (Predicted)": None,
                "Severity (GT)": s_gt, "Severity (Predicted)": s_p,
                "Confidence (GT)": "High", "Confidence (Predicted)": "High",
                "Visible Symptoms (GT)": "spots", "Visible Symptoms (Predicted)": "spots",
                "Reasoning (GT)": "spots", "Reasoning (Predicted)": "spots"}
    df = pd.DataFrame([
        row("Healthy", None, "Healthy", None),                                   # correct: no disease
        row("Not Healthy", "Nutrient Disease", "Not Healthy", "Nutrient", "Low", "Low"),
        row("Not Healthy", "Abiotic", "Healthy", None, "Low", None),            # missed
        row("Healthy", None, "Not Healthy", "Viral", None, "High"),             # false alarm
    ])
    summary, _ = ev.evaluate_model(df)
    assert summary["health_accuracy"] == pytest.approx(50.0)
    assert summary["category_accuracy"] == pytest.approx(50.0)
    assert summary["severity_accuracy"] == pytest.approx(50.0)
    assert summary["confidence_accuracy"] == pytest.approx(100.0)
