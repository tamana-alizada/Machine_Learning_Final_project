"""FastAPI deployment for the Drug Allergy ML model.

Run locally:
    uvicorn main:app --reload

Open docs:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_utils import DEFAULT_INPUT, load_model_bundle, predict_records, rows_to_dataframe


class DrugAllergyInput(BaseModel):
    age_years: float = Field(30, ge=0, le=120, description="Patient age in years")
    gender: str = Field("unknown", description="Patient gender text, e.g. male, female, unknown")
    gender_code: int = Field(0, description="Encoded gender value if available")
    weight_kg: float = Field(70, ge=0, le=300, description="Patient weight in kilograms")
    drug_name: str = Field(..., min_length=1, description="Drug name, e.g. aspirin")
    drug_role: str = Field("suspect", description="Drug role, e.g. suspect, concomitant")
    active_ingredients: str = Field("unknown", description="Active ingredients of the drug")
    reactions: str = Field("unknown", description="Observed or reported reactions")
    previous_allergic_reactions: str = Field("none", description="Previous allergy reaction text")
    has_previous_allergy: int = Field(0, ge=0, le=1, description="1 if patient has previous allergy, else 0")
    is_serious: int = Field(0, ge=0, le=1, description="1 if report/event is serious, else 0")


class BatchDrugAllergyInput(BaseModel):
    items: list[DrugAllergyInput]


class PredictionResponse(BaseModel):
    row: int = 1
    prediction: int
    allergic_to_drug: bool
    risk_label: str
    allergy_probability: float
    threshold: float
    model_name: str
    explanation: list[dict[str, Any]]


app = FastAPI(
    title="Personalized Drug Allergy Prediction API",
    description="FastAPI service that loads a saved joblib ML model and returns allergy-risk predictions with explanations.",
    version="2.0.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ML API is running"}


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        bundle = load_model_bundle()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_name": bundle.get("model_name"),
            "threshold": bundle.get("threshold"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    try:
        bundle = load_model_bundle()
        return {
            "model_name": bundle.get("model_name"),
            "threshold": bundle.get("threshold"),
            "use_reaction_features": bundle.get("use_reaction_features"),
            "columns": bundle.get("columns"),
            "example_input": DEFAULT_INPUT,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(item: DrugAllergyInput) -> PredictionResponse:
    try:
        result = predict_records(item.model_dump(), include_explanations=True)[0]
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(batch: BatchDrugAllergyInput) -> list[PredictionResponse]:
    try:
        rows = [item.model_dump() for item in batch.items]
        return [PredictionResponse(**result) for result in predict_records(rows, include_explanations=True)]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc


@app.post("/explain")
def explain(item: DrugAllergyInput) -> dict[str, Any]:
    """Return only the model explanation for one input."""
    try:
        result = predict_records(item.model_dump(), include_explanations=True)[0]
        return {
            "risk_label": result["risk_label"],
            "allergy_probability": result["allergy_probability"],
            "threshold": result["threshold"],
            "explanation": result["explanation"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}") from exc


@app.post("/validate-input")
def validate_input(item: DrugAllergyInput) -> dict[str, Any]:
    """Debug helper: show the exact row shape that goes into the model."""
    df = rows_to_dataframe(item.model_dump())
    return {"columns": list(df.columns), "row": df.iloc[0].to_dict()}
