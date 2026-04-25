"""
main.py
=======
FastAPI server for Cancer Relapse Prediction.
Provides /predict endpoint and /model-info for frontend.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(
    title="Cancer Relapse Prediction API",
    description="AI-powered API for predicting cancer relapse risk based on clinical patient data",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and evaluation data
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'combined_cancer_data.csv')
model_path = os.path.join(MODEL_DIR, 'model_pipeline.pkl')
eval_path = os.path.join(MODEL_DIR, 'evaluation_results.json')
imp_path = os.path.join(MODEL_DIR, 'feature_importances.json')

pipeline = None
eval_results = None
feature_importances = None

try:
    pipeline = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    with open(eval_path, 'r') as f:
        eval_results = json.load(f)
except Exception:
    eval_results = {}

try:
    with open(imp_path, 'r') as f:
        feature_importances = json.load(f)
except Exception:
    feature_importances = {}


class PatientData(BaseModel):
    Age: int = Field(..., ge=18, le=100, description="Patient age")
    Sex: str = Field(..., description="Male or Female")
    Athleticity: str = Field(..., description="Low, Medium, or High")
    BMI: float = Field(..., ge=10, le=60, description="Body Mass Index")
    Smoking_Alcohol_History: str = Field(..., description="None, Occasional, Frequent, Heavy")
    Cancer_Type: str = Field(..., description="Breast, Lung, Colon, Prostate, Liver, Mouth")
    Tumor_Stage: int = Field(..., ge=1, le=4, description="Tumor stage 1-4")
    Tumor_Grade: int = Field(..., ge=1, le=3, description="Tumor grade 1-3")
    Tumor_Size_cm: float = Field(..., ge=0.1, le=30, description="Tumor size in cm")
    Lymph_Nodes_Involved: str = Field(..., description="Yes or No")
    Metastasis: str = Field(..., description="Yes or No")
    Tumor_Type: str = Field(..., description="Malignant or Benign")
    Hormone_Receptor: str = Field(..., description="Positive, Negative, or Not Applicable")
    Gene_Mutations: str = Field(..., description="None, TP53, BRCA1/2, Other")
    Surgery_Type: str = Field(..., description="Surgery type performed")
    Chemotherapy: str = Field(..., description="Yes or No")
    Radiation_Therapy: str = Field(..., description="Yes or No")
    Hormone_Therapy: str = Field(..., description="Yes or No")
    Immunotherapy: str = Field(..., description="Yes or No")
    Time_Since_Treatment_Months: int = Field(..., ge=0, le=240, description="Months since treatment ended")
    Follow_Up_Visits: int = Field(..., ge=0, le=100, description="Number of follow-up visits")
    Previous_Reoccurrence: str = Field(..., description="Yes or No")


@app.get("/")
def read_root():
    return {"message": "Cancer Relapse Prediction API is running.", "version": "1.0.0"}


@app.post("/predict")
def predict_relapse(data: PatientData):
    import traceback
    if not pipeline:
        return {"error": "Model not loaded on the server."}

    try:
        df = pd.DataFrame([data.model_dump()])
        print(f"Input DataFrame columns: {list(df.columns)}")
        print(f"Input DataFrame dtypes:\n{df.dtypes}")
        prediction = pipeline.predict(df)[0]

        try:
            probability = pipeline.predict_proba(df)[0][1] * 100
        except Exception:
            probability = None

        # Determine risk factors
        risk_factors = []
        if data.Tumor_Stage >= 3:
            risk_factors.append("Advanced tumor stage (Stage {})".format(data.Tumor_Stage))
        if data.Lymph_Nodes_Involved == 'Yes':
            risk_factors.append("Lymph node involvement detected")
        if data.Metastasis == 'Yes':
            risk_factors.append("Metastasis present")
        if data.Smoking_Alcohol_History in ['Frequent', 'Heavy']:
            risk_factors.append("Significant smoking/alcohol history")
        if data.Tumor_Size_cm > 5:
            risk_factors.append("Large tumor size (>{:.1f} cm)".format(data.Tumor_Size_cm))
        if data.Previous_Reoccurrence == 'Yes':
            risk_factors.append("Previous reoccurrence history")
        if data.Tumor_Grade == 3:
            risk_factors.append("High-grade tumor (Grade 3)")
        if data.BMI > 35:
            risk_factors.append("Elevated BMI")

        # Include model stats
        best_model_name = eval_results.get("best_model", "Unknown")
        model_stats = eval_results.get("all_results", {}).get(best_model_name, {})

        return {
            "prediction": "Yes" if prediction == 1 else "No",
            "probability_percentage": round(probability, 2) if probability is not None else None,
            "risk_factors": risk_factors,
            "risk_level": "High" if (probability and probability > 60) else "Medium" if (probability and probability > 35) else "Low",
            "model_stats": {
                "name": best_model_name,
                "accuracy": model_stats.get("accuracy", 0) * 100,
                "recall": model_stats.get("recall", 0) * 100,
                "precision": model_stats.get("precision", 0) * 100,
                "roc_auc": model_stats.get("roc_auc", 0) * 100
            }
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/model-info")
def model_info():
    return {
        "model_name": eval_results.get("best_model", "Unknown"),
        "metrics": eval_results.get("all_results", {}),
        "dataset_size": eval_results.get("dataset_size", 0),
        "feature_importances": feature_importances or {}
    }


@app.get("/visualizations")
def get_visualizations():
    """Return aggregated chart data for the frontend dashboard."""
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        return {"error": f"Could not load data: {e}"}

    df['Relapse_Binary'] = (df['Relapse'] == 'Yes').astype(int)

    # 1. Relapse rate by cancer type
    cancer_relapse = df.groupby('Cancer_Type')['Relapse_Binary'].agg(['mean', 'count']).reset_index()
    cancer_relapse.columns = ['cancer_type', 'relapse_rate', 'count']
    cancer_relapse['relapse_rate'] = (cancer_relapse['relapse_rate'] * 100).round(1)

    # 2. Age distribution by relapse
    age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=[f"{a}-{b-1}" for a, b in zip(age_bins[:-1], age_bins[1:])])
    age_dist = df.groupby(['Age_Group', 'Relapse']).size().unstack(fill_value=0).reset_index()
    age_data = []
    for _, row in age_dist.iterrows():
        age_data.append({
            "age_group": str(row['Age_Group']),
            "no_relapse": int(row.get('No', 0)),
            "relapse": int(row.get('Yes', 0))
        })

    # 3. Tumor stage vs relapse rate
    stage_relapse = df.groupby('Tumor_Stage')['Relapse_Binary'].mean().reset_index()
    stage_data = [{"stage": f"Stage {int(r['Tumor_Stage'])}", "relapse_rate": round(r['Relapse_Binary'] * 100, 1)}
                  for _, r in stage_relapse.iterrows()]

    # 4. Treatment impact on relapse
    treatments = ['Chemotherapy', 'Radiation_Therapy', 'Hormone_Therapy', 'Immunotherapy']
    treatment_data = []
    for t in treatments:
        yes_rate = df[df[t] == 'Yes']['Relapse_Binary'].mean() * 100
        no_rate = df[df[t] == 'No']['Relapse_Binary'].mean() * 100
        treatment_data.append({
            "treatment": t.replace('_', ' '),
            "with_treatment": round(yes_rate, 1),
            "without_treatment": round(no_rate, 1)
        })

    # 5. Smoking impact
    smoking_relapse = df.groupby('Smoking_Alcohol_History')['Relapse_Binary'].mean().reset_index()
    smoking_order = ['None', 'Occasional', 'Frequent', 'Heavy']
    smoking_data = []
    for level in smoking_order:
        row = smoking_relapse[smoking_relapse['Smoking_Alcohol_History'] == level]
        if len(row) > 0:
            smoking_data.append({"level": level, "relapse_rate": round(row.iloc[0]['Relapse_Binary'] * 100, 1)})

    # 6. Model comparison
    model_data = []
    if eval_results:
        for name, metrics in eval_results.get("all_results", {}).items():
            model_data.append({
                "model": name,
                "accuracy": metrics.get("accuracy", 0) * 100,
                "recall": metrics.get("recall", 0) * 100,
                "precision": metrics.get("precision", 0) * 100,
                "f1": metrics.get("f1_score", 0) * 100,
                "roc_auc": metrics.get("roc_auc", 0) * 100
            })

    # 7. Dataset summary
    summary = {
        "total_records": len(df),
        "relapse_count": int(df['Relapse_Binary'].sum()),
        "no_relapse_count": int((df['Relapse_Binary'] == 0).sum()),
        "cancer_types": int(df['Cancer_Type'].nunique()),
        "real_data_pct": round(len(df[df['Data_Source'].str.contains('UCI|WPBC')]) / len(df) * 100, 1),
        "avg_age": round(df['Age'].mean(), 1),
        "best_model": eval_results.get("best_model", "Unknown")
    }

    return {
        "cancer_relapse": cancer_relapse.to_dict(orient='records'),
        "age_distribution": age_data,
        "stage_relapse": stage_data,
        "treatment_impact": treatment_data,
        "smoking_impact": smoking_data,
        "model_comparison": model_data,
        "feature_importances": feature_importances or {},
        "summary": summary
    }
