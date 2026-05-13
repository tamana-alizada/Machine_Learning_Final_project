# Drug Allergy Risk Prediction System

A complete machine learning deployment project for predicting possible drug allergy risk using patient, drug, and reaction-related information.

This project includes model training, model evaluation, saved model deployment, FastAPI backend, Streamlit frontend, Docker support, MLflow tracking, and simple model explanation.

> **Important:** This project is for academic/class use only. It is not a medical diagnosis system and must not be used as real medical advice.

---

## 1. Project Overview

Drug allergy information in clinical records can sometimes be incomplete, outdated, or written as free text. This can make it harder for doctors or pharmacists to quickly identify possible allergy risks.

This project builds a machine learning-based decision support system that predicts possible drug allergy risk and presents the result in a user-friendly interface.

The goal is not to replace doctors. The goal is to support safer medication review by giving an early warning signal when allergy risk may exist.

---

## 2. Problem Statement

Medication safety is an important issue in healthcare. If allergy information is missing, unclear, or not reviewed properly, a patient may receive a drug that can cause a harmful reaction.

Many clinical records contain allergy information in different forms. Some records may be structured, while others may be written as free text. This makes manual checking slower and less reliable.

This project addresses this problem by developing a machine learning system that receives patient and drug-related input and predicts possible allergy risk.

---

## 3. Project Objectives

The main objectives of this project are:

- To build a machine learning model for drug allergy risk prediction
- To compare multiple machine learning algorithms
- To handle class imbalance during model training
- To focus on recall because missing real allergy cases can be dangerous
- To deploy the trained model using FastAPI
- To create a user-friendly Streamlit frontend
- To support single and batch predictions
- To provide simple model explanations
- To use Docker for easier deployment
- To use MLflow for experiment tracking and model registry support

---

## 4. Main Features

- Machine learning model for drug allergy risk prediction
- FastAPI backend for API-based prediction
- Streamlit frontend for easy user interaction
- Single patient prediction
- Multiple manual input prediction
- Batch prediction using CSV upload
- Model explanation for each prediction
- Input validation
- Docker deployment
- MLflow experiment tracking
- MLflow model registry support
- Monitoring and drift-checking idea for future model quality control

---

## 5. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- FastAPI
- Streamlit
- MLflow
- Docker
- Joblib
- Uvicorn

---

## 6. Machine Learning Approach

The project tests different machine learning models, including:

- Logistic Regression
- Random Forest
- LightGBM Gradient Boosting

The final selected model is saved as:

```text
model.joblib
```

The model uses features such as:

- Age
- Gender
- Weight
- Drug name
- Drug role
- Active ingredients
- Observed reactions
- Previous allergic reactions
- Previous allergy history
- Serious reaction status

Because this is a medical safety-related task, the project gives strong attention to **recall**. High recall is important because missing a real allergy case can be dangerous.

A false positive may cause extra checking, but a false negative may allow a risky drug to be given to a patient. For this reason, recall is treated as one of the most important evaluation goals.

---

## 7. Dataset Information

The project uses a prepared drug allergy dataset based on adverse event-style information.

The dataset includes information such as:

- Patient age
- Patient gender
- Patient weight
- Drug name
- Drug role
- Active ingredients
- Reaction terms
- Previous allergic reactions
- Previous allergy status
- Serious reaction status

The dataset is used for academic learning and machine learning system development.

---

## 8. Important Data Limitation

Adverse event data can show that a reaction was reported after a drug was used. However, this does not always prove that the drug caused the reaction.

For this reason, the model output should be understood as a **risk-support signal**, not a final medical diagnosis.

The system should only support doctors, pharmacists, or medical staff in reviewing possible allergy risk.

---

## 9. Project Structure

```text
ml-fastapi-docker-drug-allergy-complete/
├── main.py                         # FastAPI backend
├── streamlit_app.py                # Streamlit frontend
├── model_utils.py                  # Shared model loading, prediction, and explanation logic
├── train.py                        # Model training and MLflow tracking
├── model.joblib                    # Saved trained model
├── requirements.txt                # Python dependencies
├── Dockerfile                      # FastAPI Docker image
├── Dockerfile.streamlit            # Streamlit Docker image
├── docker-compose.yml              # Runs backend and frontend together
├── sample_request.json             # Example single prediction request
├── sample_batch_request.json       # Example batch API request
├── sample_batch_input.csv          # Example CSV file for batch prediction
├── test_api.py                     # API testing script
├── notebooks/                      # Development and experiment notebooks
└── README.md
```

---

## 10. Installation

### Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recommended Python version:

```text
Python 3.10 or 3.11
```

---

## 11. Run the FastAPI Backend

Start the backend:

```bash
uvicorn main:app --reload
```

Open in browser:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 12. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Check if the API is running |
| GET | `/health` | Check if the model is loaded |
| GET | `/model-info` | Show model information |
| POST | `/predict` | Predict one input |
| POST | `/predict/batch` | Predict many inputs |
| POST | `/explain` | Explain prediction |
| POST | `/validate-input` | Validate input before prediction |

Test the API:

```bash
python test_api.py
```

---

## 13. Run the Streamlit Frontend

Start the frontend:

```bash
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The Streamlit interface supports:

- Single input prediction
- Multiple manual inputs
- CSV upload
- Batch prediction
- Downloading prediction results
- Showing model-based explanation

---

## 14. Run with Docker

Docker allows the project to run the same way on different computers without manually installing all dependencies.

### Run both FastAPI and Streamlit together

```bash
docker compose up --build
```

Open:

```text
FastAPI docs: http://127.0.0.1:8000/docs
Streamlit UI: http://127.0.0.1:8501
```

### Stop containers

```bash
docker compose down
```

---

## 15. Run FastAPI Only with Docker

Build the FastAPI image:

```bash
docker build -t drug-allergy-api .
```

Run the FastAPI container:

```bash
docker run -p 8000:8000 drug-allergy-api
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## 16. Run Streamlit Only with Docker

Build the Streamlit image:

```bash
docker build -f Dockerfile.streamlit -t drug-allergy-streamlit .
```

Run the Streamlit container:

```bash
docker run -p 8501:8501 drug-allergy-streamlit
```

Open:

```text
http://127.0.0.1:8501
```

---

## 17. Example Single Prediction Request

```json
{
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
  "is_serious": 0
}
```

---

## 18. Model Explanation

The system provides a simple explanation for each prediction.

For tree-based models such as Random Forest and LightGBM, the explanation is based on:

```text
feature importance × active input value
```

For linear models, the explanation is based on:

```text
coefficient × active input value
```

This helps the user understand which factors influenced the model result, such as reaction terms, drug name, active ingredients, or patient-related features.

The explanation is not a medical explanation. It only shows which input features influenced the machine learning model.

---

## 19. Train the Model Again

If the dataset is available in the project folder:

```text
openfda_drug_allergy_dataset.csv
```

Run:

```bash
python train.py
```

Or give the dataset path manually:

```bash
python train.py --data openfda_drug_allergy_dataset.csv --output model.joblib
```

The training script performs:

- Data loading
- Data preprocessing
- Model training
- Cross-validation
- Class imbalance handling
- Threshold selection based on recall
- Test evaluation
- Saving the trained model
- Saving result files
- MLflow experiment logging

To train without MLflow:

```bash
python train.py --skip-mlflow
```

---

## 20. MLflow Tracking

Start the MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open:

```text
http://127.0.0.1:5000
```

MLflow is used to track:

- Model parameters
- Evaluation metrics
- Training runs
- Model artifacts
- Registered model versions

The registered model name is:

```text
DrugAllergyRiskModel
```

---

## 21. Model Monitoring and Quality Assurance

The project includes a monitoring idea for checking model quality after deployment.

The system can monitor:

- Model performance on new labeled data
- Recall
- False negatives
- Feature drift
- Changes in drug names
- Changes in reaction text
- Changes in active ingredients

Feature drift means that new input data becomes different from the original training data.

For example, if new drug names, new active ingredients, or new reaction terms appear often, the model may need to be reviewed. If performance becomes worse, the model should be updated or retrained.

---

## 22. Business and Hospital Value

From a hospital or business perspective, this project can provide value by supporting:

- Improved medication safety
- Faster review of allergy history
- Better allergy documentation
- More structured collection of outcomes
- Future model improvement using organized feedback

Better documentation means allergy information is recorded clearly and consistently. This is important because poor documentation can cause doctors to miss important allergy warnings in the future.

---

## 23. Scientific Value

The project has scientific value because it tests machine learning for drug allergy risk prediction and shows how explainable prediction can be used in a decision-support system.

It also clearly explains the limitations of the data. The project does not claim that the model proves medical causation. It only provides a risk signal that should be reviewed by a medical professional.

---

## 24. Limitations

This project has important limitations:

- The dataset is based on adverse event-style reports.
- Reported adverse events do not prove that a drug caused the allergy.
- The model prediction is only a risk-support signal.
- The system is not clinically validated in real hospitals.
- The system should not be used as a real diagnosis tool.
- A doctor or pharmacist must review the final decision.
- The system may need retraining if new drug or reaction patterns appear.

---

## 25. Future Work

Future improvements may include:

- Testing the model with real hospital data
- Multicenter clinical validation
- Better calibration checks
- SHAP-based explanations
- Stronger privacy and security review
- Integration with real Electronic Health Record systems
- Continuous monitoring
- Automatic retraining workflow

---

## 26. Academic Purpose

This project was developed as a class project to demonstrate a complete machine learning system.

It covers:

- Data preparation
- Model training
- Model evaluation
- API development
- Frontend development
- Deployment using Docker
- MLflow tracking
- User-facing decision support

This is a learning project and should not be used for real medical treatment decisions.

---

## 27. Prepared by

Alizada Tamana
Azizi Hashmatullah
