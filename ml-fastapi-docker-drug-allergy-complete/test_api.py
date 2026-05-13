"""Small local API test.

First run the server:
    uvicorn main:app --reload

Then run:
    python test_api.py
"""

import json

import requests

BASE_URL = "http://127.0.0.1:8000"

sample = {
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

batch = {"items": [sample, {**sample, "drug_name": "ibuprofen", "reactions": "nausea headache", "has_previous_allergy": 0}]}

root = requests.get(f"{BASE_URL}/", timeout=10)
print("GET /", root.status_code, root.json())

health = requests.get(f"{BASE_URL}/health", timeout=10)
print("GET /health", health.status_code, health.json())

prediction = requests.post(f"{BASE_URL}/predict", json=sample, timeout=10)
print("POST /predict", prediction.status_code)
print(json.dumps(prediction.json(), indent=2))

batch_prediction = requests.post(f"{BASE_URL}/predict/batch", json=batch, timeout=10)
print("POST /predict/batch", batch_prediction.status_code)
print(json.dumps(batch_prediction.json(), indent=2))
