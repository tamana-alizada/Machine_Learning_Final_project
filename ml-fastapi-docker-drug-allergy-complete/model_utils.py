"""Shared model loading, prediction, and explanation helpers.

These helpers are used by both FastAPI (main.py) and Streamlit
(streamlit_app.py), so the API and UI always behave the same way.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).with_name("model.joblib")

DEFAULT_INPUT = {
    "age_years": 35,
    "gender": "female",
    "gender_code": 1,
    "weight_kg": 65,
    "drug_name": "aspirin",
    "drug_role": "suspect",
    "active_ingredients": "acetylsalicylic acid",
    "reactions": "rash itching swelling",
    "previous_allergic_reactions": "rash",
    "has_previous_allergy": 1,
    "is_serious": 0,
}

TEXT_COLUMNS = [
    "gender",
    "drug_name",
    "drug_role",
    "active_ingredients",
    "reactions",
    "previous_allergic_reactions",
]

NUMERIC_COLUMNS = [
    "age_years",
    "gender_code",
    "weight_kg",
    "has_previous_allergy",
    "is_serious",
]


def _patch_sklearn_private_class() -> None:
    """Best-effort compatibility patch for sklearn private pickle class changes."""
    try:
        import sklearn.compose._column_transformer as ct

        if not hasattr(ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            ct._RemainderColsList = _RemainderColsList
    except Exception:
        pass


def _iter_estimators(obj: Any):
    """Yield nested sklearn-style estimators inside pipelines/column transformers."""
    seen = set()

    def visit(x: Any):
        ident = id(x)
        if ident in seen:
            return
        seen.add(ident)
        yield x
        if hasattr(x, "steps"):
            for _, step in x.steps:
                yield from visit(step)
        if hasattr(x, "transformers"):
            for transformer in x.transformers:
                if len(transformer) >= 2 and transformer[1] not in ("drop", "passthrough"):
                    yield from visit(transformer[1])
        if hasattr(x, "transformers_"):
            for transformer in x.transformers_:
                if len(transformer) >= 2 and transformer[1] not in ("drop", "passthrough"):
                    yield from visit(transformer[1])

    yield from visit(obj)


def _patch_loaded_model_for_newer_sklearn(model: Any) -> None:
    """Best-effort compatibility when running outside pinned requirements."""
    try:
        from sklearn.impute import SimpleImputer
    except Exception:
        SimpleImputer = None

    for estimator in _iter_estimators(model):
        if SimpleImputer is not None and isinstance(estimator, SimpleImputer):
            if not hasattr(estimator, "_fill_dtype"):
                stats = getattr(estimator, "statistics_", None)
                estimator._fill_dtype = getattr(stats, "dtype", None) or np.float64


@lru_cache(maxsize=1)
def load_model_bundle(model_path: Path | str = MODEL_PATH) -> dict[str, Any]:
    """Load the saved joblib model bundle once and cache it."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    _patch_sklearn_private_class()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(path)

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("model.joblib must contain a dictionary with key 'model'.")

    bundle.setdefault("threshold", 0.5)
    bundle.setdefault("model_name", type(bundle["model"]).__name__)
    bundle.setdefault("columns", list(DEFAULT_INPUT.keys()))
    bundle.setdefault("use_reaction_features", "reactions" in bundle.get("columns", []))
    _patch_loaded_model_for_newer_sklearn(bundle["model"])
    return bundle


def clean_text(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    return text if text else "unknown"


def normalize_row(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """Clean one input row and force it to match the training column order."""
    data = {**DEFAULT_INPUT, **dict(row)}

    for col in TEXT_COLUMNS:
        if col in data:
            data[col] = clean_text(data[col])

    for col in NUMERIC_COLUMNS:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    normalized = {col: data.get(col, DEFAULT_INPUT.get(col, "unknown")) for col in columns}
    return normalized


def rows_to_dataframe(rows: list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Convert one row or many rows into a model-ready DataFrame."""
    bundle = load_model_bundle()
    columns = list(bundle.get("columns", DEFAULT_INPUT.keys()))

    if isinstance(rows, dict):
        rows = [rows]

    normalized_rows = [normalize_row(row, columns) for row in rows]
    return pd.DataFrame(normalized_rows, columns=columns)


def get_probabilities(model: Any, df: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(df))[:, 1].astype(float)
        return np.asarray(model.predict(df)).astype(float)


def risk_label_from_probability(probability: float, threshold: float) -> str:
    return "High allergy risk" if probability >= threshold else "Low allergy risk"


def prettify_feature_name(raw_name: str) -> str:
    """Turn pipeline feature names into friendly UI text."""
    name = raw_name
    if "__" in name:
        prefix, rest = name.split("__", 1)
    else:
        prefix, rest = "feature", name

    rest = rest.replace("_", " ").strip()
    prefix_lower = prefix.lower()

    if "drug_name" in prefix_lower:
        return f"Drug name contains: {rest}"
    if "active_ingredients" in prefix_lower:
        return f"Active ingredient contains: {rest}"
    if "reactions" in prefix_lower:
        return f"Reaction text contains: {rest}"
    if prefix_lower.startswith("cat"):
        return f"Category: {rest}"
    if prefix_lower.startswith("num"):
        return f"Numeric value: {rest}"
    return rest.title()


def _get_feature_values(model: Any, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if not hasattr(model, "named_steps") or "prep" not in model.named_steps or "clf" not in model.named_steps:
        return None, None

    prep = model.named_steps["prep"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transformed = prep.transform(df)
    feature_names = prep.get_feature_names_out()

    if hasattr(transformed, "toarray"):
        values = transformed.toarray()[0]
    else:
        values = np.asarray(transformed)[0]

    return np.asarray(feature_names), np.asarray(values)


def explain_prediction(model: Any, df: pd.DataFrame, top_n: int = 8) -> list[dict[str, Any]]:
    """Create a simple model-based local explanation for one row.

    For linear models, score = coefficient * transformed input value.
    For tree/boosting models, score = feature importance * active input value.

    This is a practical class-project explanation. It shows which input signals the
    model relied on most. It is not a medical or causal explanation.
    """
    try:
        feature_names, values = _get_feature_values(model, df.iloc[[0]])
        if feature_names is None:
            return []

        clf = model.named_steps["clf"]
        explanation_type = "model_importance"

        if hasattr(clf, "coef_"):
            raw_scores = np.asarray(clf.coef_[0]) * values
            explanation_type = "linear_contribution"
        elif hasattr(clf, "feature_importances_"):
            raw_scores = np.asarray(clf.feature_importances_) * np.abs(values)
        else:
            return []

        active_idx = np.where(np.abs(raw_scores) > 0)[0]
        if len(active_idx) == 0:
            active_idx = np.argsort(np.abs(raw_scores))[-top_n:]

        ranked_idx = sorted(active_idx, key=lambda i: abs(raw_scores[i]), reverse=True)[:top_n]
        max_abs = max(float(np.max(np.abs(raw_scores[ranked_idx]))), 1e-12)

        rows = []
        for idx in ranked_idx:
            score = float(raw_scores[idx])
            if explanation_type == "linear_contribution":
                direction = "pushes risk up" if score > 0 else "pushes risk down"
            else:
                direction = "important signal used by the model"

            rows.append(
                {
                    "feature": str(feature_names[idx]),
                    "factor": prettify_feature_name(str(feature_names[idx])),
                    "score": round(score, 6),
                    "importance_percent": round(abs(score) / max_abs * 100, 2),
                    "direction": direction,
                    "explanation_type": explanation_type,
                }
            )
        return rows
    except Exception:
        return []


def predict_records(rows: list[dict[str, Any]] | dict[str, Any], include_explanations: bool = True) -> list[dict[str, Any]]:
    """Predict one or many records and return friendly dictionaries."""
    bundle = load_model_bundle()
    model = bundle["model"]
    threshold = float(bundle.get("threshold", 0.5))
    model_name = str(bundle.get("model_name", "unknown"))
    df = rows_to_dataframe(rows)
    probabilities = get_probabilities(model, df)

    results: list[dict[str, Any]] = []
    for idx, probability in enumerate(probabilities):
        pred = int(probability >= threshold)
        row_result = {
            "row": idx + 1,
            "prediction": pred,
            "allergic_to_drug": bool(pred),
            "risk_label": risk_label_from_probability(float(probability), threshold),
            "allergy_probability": round(float(probability), 6),
            "threshold": threshold,
            "model_name": model_name,
        }
        if include_explanations:
            row_result["explanation"] = explain_prediction(model, df.iloc[[idx]])
        results.append(row_result)

    return results
