"""Training script for the Drug Allergy Prediction project.

This version is aligned with the executed notebook:
- Same train / validation / test split.
- Same preprocessing.
- Same model parameters.
- Same threshold tuning rule.
- Same best-model selection score: 0.65 * recall + 0.35 * accuracy.

Run:
    python train.py --data openfda_drug_allergy_dataset.csv

Or, if the CSV is in the same folder:
    python train.py

Output:
    model.joblib
    model_validation_comparison.csv
    model_cv_results.csv
    model_test_metrics.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "allergic_to_drug"
USE_REACTION_FEATURES = True
MLFLOW_EXPERIMENT_NAME = "Drug Allergy ML Pipeline"
REGISTERED_MODEL_NAME = "DrugAllergyRiskModel"


# -----------------------------------------------------------------------------
# Data loading and cleaning
# -----------------------------------------------------------------------------

def find_dataset_path(user_path: str | None) -> Path:
    """Find the dataset path.

    The old train.py required --data every time. This version can also find the
    common dataset filenames automatically when the file is in the project folder.
    """
    possible_paths = []

    if user_path:
        possible_paths.append(Path(user_path))

    possible_paths.extend(
        [
            Path("openfda_drug_allergy_dataset.csv"),
            Path("openfda_drug_allergy_dataset(1).csv"),
            Path("data/openfda_drug_allergy_dataset.csv"),
            Path("data/openfda_drug_allergy_dataset(1).csv"),
            Path("/content/openfda_drug_allergy_dataset.csv"),
            Path("/content/openfda_drug_allergy_dataset(1).csv"),
            Path("/mnt/data/openfda_drug_allergy_dataset.csv"),
            Path("/mnt/data/openfda_drug_allergy_dataset(1).csv"),
        ]
    )

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "CSV file not found. Put openfda_drug_allergy_dataset.csv in this folder "
        "or run: python train.py --data path/to/your_dataset.csv"
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset in the same way as the notebook."""
    df = df.copy()

    # Normalize text columns.
    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        df[col] = df[col].fillna("unknown").astype(str).str.lower().str.strip()

    # Keep numeric columns numeric.
    numeric_cols = [
        "age_years",
        "weight_kg",
        "gender_code",
        "has_previous_allergy",
        "is_serious",
        TARGET,
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# -----------------------------------------------------------------------------
# Preprocessing and models
# -----------------------------------------------------------------------------

def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    numeric_features = [
        "age_years",
        "weight_kg",
        "gender_code",
        "has_previous_allergy",
        "is_serious",
    ]

    categorical_features = [
        "gender",
        "drug_role",
        "previous_allergic_reactions",
    ]

    text_features = ["drug_name", "active_ingredients"]
    if USE_REACTION_FEATURES:
        text_features.append("reactions")

    # Keep only columns that actually exist in the CSV.
    numeric_features = [c for c in numeric_features if c in X.columns]
    categorical_features = [c for c in categorical_features if c in X.columns]
    text_features = [c for c in text_features if c in X.columns]

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                categorical_features,
            )
        )

    tfidf_limits = {
        "drug_name": 100,
        "active_ingredients": 100,
        "reactions": 200,
    }

    for col in text_features:
        transformers.append(
            (
                f"{col}_tfidf",
                TfidfVectorizer(
                    max_features=tfidf_limits.get(col, 100),
                    ngram_range=(1, 2),
                    min_df=2,
                ),
                col,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns found in the dataset.")

    preprocessor = ColumnTransformer(transformers, sparse_threshold=0.7)
    return preprocessor, numeric_features, categorical_features, text_features


def build_models(preprocessor: ColumnTransformer) -> dict[str, object]:
    """Build the same models used in the notebook.

    Important fix:
    The previous train.py used different RandomForest settings and used both
    class_weight and scale_pos_weight for LightGBM. This version matches the
    notebook and removes duplicate positive-class weighting from LightGBM.
    """
    return {
        "Logistic Regression - class_weight": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="liblinear",
                        C=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Logistic Regression - oversampling": ImbPipeline(
            [
                ("prep", preprocessor),
                ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        solver="liblinear",
                        C=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest - class_weight": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=80,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "LightGBM Gradient Boosting - class_weight": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=50,
                        num_leaves=15,
                        learning_rate=0.10,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                        verbose=-1,
                        force_col_wise=True,
                    ),
                ),
            ]
        ),
    }


# -----------------------------------------------------------------------------
# Evaluation and threshold tuning
# -----------------------------------------------------------------------------

def safe_roc_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def safe_pr_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def evaluate_at_threshold(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
    }


def tune_threshold(y_true, y_prob, min_recall: float = 0.80) -> tuple[float, pd.DataFrame]:
    """Tune threshold exactly like the notebook.

    First, keep thresholds with recall >= 0.80.
    From those, choose the one with best accuracy, then F2, then precision.
    If no threshold reaches 0.80 recall, choose best F2, then accuracy.
    """
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        rows.append(evaluate_at_threshold(y_true, y_prob, float(threshold)))

    curve = pd.DataFrame(rows)
    good = curve[curve["recall"] >= min_recall]

    if len(good) > 0:
        best = good.sort_values(["accuracy", "f2", "precision"], ascending=False).iloc[0]
    else:
        best = curve.sort_values(["f2", "accuracy"], ascending=False).iloc[0]

    return float(best["threshold"]), curve


# -----------------------------------------------------------------------------
# MLflow logging and model registry
# -----------------------------------------------------------------------------

def _log_numeric_metrics(prefix: str, metrics: dict) -> None:
    """Log only finite numeric values to MLflow."""
    import mlflow

    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            value = float(value)
            if np.isfinite(value):
                mlflow.log_metric(f"{prefix}_{key}", value)


def log_to_mlflow(
    *,
    final_model,
    output_path: str,
    best_model_name: str,
    best_threshold: float,
    test_metrics: dict,
    validation_results: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    text_features: list[str],
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
) -> None:
    """Log the run to MLflow and register the final model."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("\nMLflow is not installed. Install requirements.txt or run with --skip-mlflow.")
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    artifact_files = [
        output_path,
        "model_validation_comparison.csv",
        "model_cv_results.csv",
        "model_test_metrics.csv",
    ]

    with mlflow.start_run(run_name=best_model_name) as run:
        run_id = run.info.run_id

        mlflow.log_param("target", TARGET)
        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_param("best_threshold", best_threshold)
        mlflow.log_param("use_reaction_features", USE_REACTION_FEATURES)
        mlflow.log_param("registered_model_name", registered_model_name)
        mlflow.set_tag("project", "drug_allergy_prediction")
        mlflow.set_tag("model_purpose", "high_recall_allergy_risk_screening")
        mlflow.set_tag("decision_threshold", str(best_threshold))

        _log_numeric_metrics("test", test_metrics)
        best_val_row = validation_results[validation_results["model"] == best_model_name].iloc[0].to_dict()
        _log_numeric_metrics("validation", best_val_row)

        feature_info = {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "text_features": text_features,
        }
        feature_info_path = Path("mlflow_feature_info.json")
        feature_info_path.write_text(json.dumps(feature_info, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(feature_info_path))

        for file_path in artifact_files:
            path = Path(file_path)
            if path.exists():
                mlflow.log_artifact(str(path))

        # This line registers the trained model in MLflow Model Registry.
        model_info = mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        registered_version = getattr(model_info, "registered_model_version", None)
        model_uri = f"runs:/{run_id}/model"
        run_info = {
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
            "run_id": run_id,
            "model_uri": model_uri,
            "registered_model_name": registered_model_name,
            "registered_version": registered_version,
        }
        Path("mlflow_run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
        mlflow.log_artifact("mlflow_run_info.json")

        print("\nMLflow logging complete.")
        print(f"Tracking URI: {tracking_uri}")
        print(f"Experiment: {experiment_name}")
        print(f"Run ID: {run_id}")
        print(f"Model URI: {model_uri}")
        print(f"Registered model: {registered_model_name}")
        if registered_version is not None:
            print(f"Registered version: {registered_version}")


# -----------------------------------------------------------------------------
# Main training workflow
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=None,
        help="Path to CSV dataset. Optional if openfda_drug_allergy_dataset.csv is in this folder.",
    )
    parser.add_argument(
        "--output",
        default="model.joblib",
        help="Output joblib path used by FastAPI. Default: model.joblib",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Train and save files, but do not log/register the model in MLflow.",
    )
    parser.add_argument(
        "--tracking-uri",
        default="file:./mlruns",
        help="MLflow tracking URI. Default: file:./mlruns",
    )
    parser.add_argument(
        "--experiment-name",
        default=MLFLOW_EXPERIMENT_NAME,
        help=f"MLflow experiment name. Default: {MLFLOW_EXPERIMENT_NAME}",
    )
    parser.add_argument(
        "--registered-model-name",
        default=REGISTERED_MODEL_NAME,
        help=f"MLflow registered model name. Default: {REGISTERED_MODEL_NAME}",
    )
    args = parser.parse_args()

    data_path = find_dataset_path(args.data)
    print(f"Dataset path: {data_path}")

    raw_data = pd.read_csv(data_path)
    data = clean_data(raw_data)

    if TARGET not in data.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    data = data.dropna(subset=[TARGET]).copy()
    data[TARGET] = data[TARGET].astype(int)

    X = data.drop(columns=[TARGET])
    y = data[TARGET].astype(int)

    print("Shape:", data.shape)
    print("Positive class ratio:", round(float(y.mean()), 4))

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # 0.17647 of the remaining 85% gives about 15% validation overall.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.17647,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )

    print("Train:", X_train.shape, "positive ratio:", round(float(y_train.mean()), 4))
    print("Validation:", X_val.shape, "positive ratio:", round(float(y_val.mean()), 4))
    print("Test:", X_test.shape, "positive ratio:", round(float(y_test.mean()), 4))

    preprocessor, numeric_features, categorical_features, text_features = build_preprocessor(X)
    models = build_models(preprocessor)

    # ------------------------------------------------------------------
    # K-Fold Cross-Validation on training set only
    # ------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "f2": make_scorer(fbeta_score, beta=2),
    }

    cv_rows = []
    for name, model in models.items():
        print(f"Running CV for: {name}")
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        row = {"model": name}
        for metric in scoring:
            row[f"cv_{metric}_mean"] = float(scores[f"test_{metric}"].mean())
            row[f"cv_{metric}_std"] = float(scores[f"test_{metric}"].std())
        cv_rows.append(row)

    cv_results = pd.DataFrame(cv_rows).sort_values("cv_recall_mean", ascending=False)
    cv_results.to_csv("model_cv_results.csv", index=False)

    print("\nCross-validation results:")
    print(cv_results.round(4).to_string(index=False))

    # ------------------------------------------------------------------
    # Validation threshold tuning and model selection
    # ------------------------------------------------------------------
    validation_rows = []
    thresholds = {}

    for name, model in models.items():
        print(f"\nFitting for validation: {name}")
        fitted = clone(model)
        fitted.fit(X_train, y_train)

        val_prob = fitted.predict_proba(X_val)[:, 1]
        best_threshold, _curve = tune_threshold(y_val, val_prob, min_recall=0.80)
        metrics = evaluate_at_threshold(y_val, val_prob, best_threshold)
        metrics["model"] = name

        validation_rows.append(metrics)
        thresholds[name] = best_threshold

    validation_results = pd.DataFrame(validation_rows)

    # Same best-model rule as the notebook.
    validation_results["selection_score"] = (
        0.65 * validation_results["recall"]
        + 0.35 * validation_results["accuracy"]
    )

    validation_results = validation_results.sort_values("selection_score", ascending=False)
    validation_results.to_csv("model_validation_comparison.csv", index=False)

    print("\nValidation comparison:")
    print(validation_results.round(4).to_string(index=False))

    best_model_name = validation_results.iloc[0]["model"]
    best_threshold = float(thresholds[best_model_name])

    print("\nBest model:", best_model_name)
    print("Best validation threshold:", round(best_threshold, 4))

    # ------------------------------------------------------------------
    # Final model: train on train + validation, test once on untouched test set
    # ------------------------------------------------------------------
    final_model = clone(models[best_model_name])
    final_model.fit(X_trainval, y_trainval)

    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)
    test_metrics = evaluate_at_threshold(y_test, test_prob, best_threshold)

    pd.DataFrame([test_metrics]).to_csv("model_test_metrics.csv", index=False)

    print("\nFinal test metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    print("\nClassification report:")
    print(classification_report(y_test, test_pred, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_pred))

    # Save the exact format expected by main.py / FastAPI.
    bundle = {
        "model": final_model,
        "threshold": best_threshold,
        "model_name": best_model_name,
        "use_reaction_features": USE_REACTION_FEATURES,
        "columns": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "text_features": text_features,
        "validation_results": validation_results.to_dict(orient="records"),
        "test_metrics": test_metrics,
    }

    joblib.dump(bundle, args.output)
    print(f"\nSaved model to: {args.output}")
    print("Saved: model_validation_comparison.csv")
    print("Saved: model_cv_results.csv")
    print("Saved: model_test_metrics.csv")


    if args.skip_mlflow:
        print("\nSkipped MLflow logging because --skip-mlflow was used.")
    else:
        log_to_mlflow(
            final_model=final_model,
            output_path=args.output,
            best_model_name=best_model_name,
            best_threshold=best_threshold,
            test_metrics=test_metrics,
            validation_results=validation_results,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            text_features=text_features,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            registered_model_name=args.registered_model_name,
        )


if __name__ == "__main__":
    main()
