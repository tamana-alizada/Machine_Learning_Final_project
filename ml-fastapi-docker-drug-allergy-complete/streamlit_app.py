"""Simple Streamlit frontend for the Drug Allergy ML model.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from io import StringIO
from typing import Any

import pandas as pd
import streamlit as st

from model_utils import DEFAULT_INPUT, load_model_bundle, predict_records

st.set_page_config(
    page_title="Personalized Drug Allergy Risk Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        padding: 1.2rem 1.4rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #f1f7ff 0%, #f8fbff 45%, #fff8f1 100%);
        border: 1px solid #e7eef8;
        margin-bottom: 1rem;
    }
    .main-title h1 { margin-bottom: 0.2rem; }
    .soft-card {
        padding: 1rem;
        border-radius: 18px;
        border: 1px solid #e7eef8;
        background: #ffffff;
        box-shadow: 0 6px 20px rgba(30, 41, 59, 0.06);
    }
    .risk-high {
        padding: 1rem;
        border-radius: 18px;
        border: 1px solid #fecaca;
        background: #fff7f7;
    }
    .risk-low {
        padding: 1rem;
        border-radius: 18px;
        border: 1px solid #bbf7d0;
        background: #f7fff9;
    }
    .small-muted { color: #64748b; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def gender_to_code(gender: str) -> int:
    gender = (gender or "unknown").lower()
    if gender == "female":
        return 1
    if gender == "male":
        return 2
    return 0


def show_header() -> None:
    st.markdown(
        """
        <div class="main-title">
            <h1>💊 Personalized Drug Allergy Risk Predictor</h1>
            <p class="small-muted">
                Simple ML interface for single and batch predictions, with clear model-based explanations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def single_input_form() -> dict[str, Any] | None:
    st.subheader("Single prediction")
    with st.form("single_prediction_form"):
        left, right = st.columns(2)
        with left:
            age_years = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=35.0, step=1.0)
            gender = st.selectbox("Gender", ["female", "male", "unknown"], index=0)
            weight_kg = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=65.0, step=1.0)
            has_previous_allergy = st.selectbox("Previous allergy?", [0, 1], index=1, help="1 = yes, 0 = no")
            is_serious = st.selectbox("Serious case?", [0, 1], index=0, help="1 = serious, 0 = not serious")
        with right:
            drug_name = st.text_input("Drug name", value="aspirin")
            drug_role = st.selectbox("Drug role", ["suspect", "concomitant", "interacting", "unknown"], index=0)
            active_ingredients = st.text_input("Active ingredients", value="acetylsalicylic acid")
            reactions = st.text_area("Observed reactions", value="rash itching swelling", height=90)
            previous_allergic_reactions = st.text_input("Previous allergic reactions", value="rash")

        submitted = st.form_submit_button("Predict allergy risk", use_container_width=True)

    if not submitted:
        return None

    return {
        "age_years": age_years,
        "gender": gender,
        "gender_code": gender_to_code(gender),
        "weight_kg": weight_kg,
        "drug_name": drug_name,
        "drug_role": drug_role,
        "active_ingredients": active_ingredients,
        "reactions": reactions,
        "previous_allergic_reactions": previous_allergic_reactions,
        "has_previous_allergy": has_previous_allergy,
        "is_serious": is_serious,
    }


def result_card(result: dict[str, Any]) -> None:
    high = bool(result["allergic_to_drug"])
    card_class = "risk-high" if high else "risk-low"
    icon = "⚠️" if high else "✅"
    st.markdown(
        f"""
        <div class="{card_class}">
            <h3>{icon} {result['risk_label']}</h3>
            <p class="small-muted">Probability: <b>{result['allergy_probability']:.3f}</b> | Threshold: <b>{result['threshold']:.3f}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_explanation(result: dict[str, Any]) -> None:
    explanation = result.get("explanation", [])
    st.subheader("Why did the model decide this?")
    st.caption("These are the strongest signals used by the model for this input. This is model interpretability, not medical advice.")

    if not explanation:
        st.info("No explanation was available for this model type.")
        return

    exp_df = pd.DataFrame(explanation)
    friendly = exp_df[["factor", "direction", "importance_percent", "score"]].rename(
        columns={
            "factor": "Factor",
            "direction": "Meaning",
            "importance_percent": "Relative importance (%)",
            "score": "Raw score",
        }
    )
    st.dataframe(friendly, use_container_width=True, hide_index=True)

    chart_df = friendly.set_index("Factor")[["Relative importance (%)"]].sort_values("Relative importance (%)")
    st.bar_chart(chart_df)


def show_single_mode() -> None:
    row = single_input_form()
    if row is None:
        st.info("Fill the form and press **Predict allergy risk**.")
        return

    result = predict_records(row, include_explanations=True)[0]
    result_card(result)
    c1, c2, c3 = st.columns(3)
    c1.metric("Allergy probability", f"{result['allergy_probability']:.3f}")
    c2.metric("Decision threshold", f"{result['threshold']:.3f}")
    c3.metric("Prediction", "Allergic" if result["allergic_to_drug"] else "Not allergic")
    show_explanation(result)

    st.download_button(
        "Download prediction JSON",
        data=json.dumps(result, indent=2),
        file_name="single_prediction_result.json",
        mime="application/json",
        use_container_width=True,
    )


def default_batch_frame() -> pd.DataFrame:
    rows = [
        DEFAULT_INPUT,
        {**DEFAULT_INPUT, "age_years": 51, "gender": "male", "gender_code": 2, "drug_name": "ibuprofen", "active_ingredients": "ibuprofen", "reactions": "nausea headache", "has_previous_allergy": 0},
        {**DEFAULT_INPUT, "age_years": 27, "gender": "unknown", "gender_code": 0, "drug_name": "amoxicillin", "active_ingredients": "amoxicillin", "reactions": "hives rash breathing difficulty", "is_serious": 1},
    ]
    return pd.DataFrame(rows)


def show_batch_results(input_df: pd.DataFrame, source_name: str) -> None:
    clean_df = input_df.dropna(how="all").copy()
    if clean_df.empty:
        st.warning("Please add at least one row.")
        return

    results = predict_records(clean_df.to_dict(orient="records"), include_explanations=True)
    results_df = pd.DataFrame(
        [
            {
                "row": r["row"],
                "risk_label": r["risk_label"],
                "allergy_probability": r["allergy_probability"],
                "threshold": r["threshold"],
                "prediction": r["prediction"],
            }
            for r in results
        ]
    )

    st.subheader(f"Prediction results: {source_name}")
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download results CSV",
        data=results_df.to_csv(index=False),
        file_name="batch_prediction_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    selected_row = st.selectbox("View explanation for row", results_df["row"].tolist())
    selected_result = results[int(selected_row) - 1]
    result_card(selected_result)
    show_explanation(selected_result)


def show_manual_batch_mode() -> None:
    st.subheader("Multiple predictions")
    st.caption("Edit the table, add more rows, then predict all rows together.")
    edited_df = st.data_editor(
        default_batch_frame(),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
    )
    if st.button("Predict all rows", use_container_width=True):
        show_batch_results(edited_df, "manual table")


def show_csv_mode() -> None:
    st.subheader("CSV batch prediction")
    st.caption("Upload a CSV with the same columns used during training. Missing columns will be filled with safe default values.")

    sample_csv = default_batch_frame().to_csv(index=False)
    st.download_button(
        "Download sample CSV template",
        data=sample_csv,
        file_name="drug_allergy_sample_input.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    st.write("Preview")
    st.dataframe(df.head(20), use_container_width=True)
    if st.button("Predict uploaded CSV", use_container_width=True):
        show_batch_results(df, uploaded_file.name)


def show_mlflow_help() -> None:
    st.subheader("MLflow tracking")
    st.markdown(
        """
        The training script can log experiments and register the model in MLflow.

        ```bash
        python train.py --data openfda_drug_allergy_dataset.csv --output model.joblib
        mlflow ui --backend-store-uri ./mlruns
        ```

        Then open `http://127.0.0.1:5000` to see runs, metrics, artifacts, and the registered model.
        """
    )


def main() -> None:
    show_header()

    try:
        bundle = load_model_bundle()
    except Exception as exc:
        st.error(f"Model could not be loaded: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Project menu")
        mode = st.radio("Choose mode", ["Single input", "Multiple inputs", "CSV upload", "MLflow guide"])
        st.divider()
        st.success("Model loaded")
        st.write(f"**Model:** {bundle.get('model_name', 'unknown')}")
        st.write(f"**Threshold:** {float(bundle.get('threshold', 0.5)):.3f}")
        st.write(f"**Features:** {len(bundle.get('columns', []))}")
        st.divider()
        st.warning("Class project only. Do not use this as real medical advice.")

    if mode == "Single input":
        show_single_mode()
    elif mode == "Multiple inputs":
        show_manual_batch_mode()
    elif mode == "CSV upload":
        show_csv_mode()
    else:
        show_mlflow_help()


if __name__ == "__main__":
    main()
