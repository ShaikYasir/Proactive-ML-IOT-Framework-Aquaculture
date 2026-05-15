

try:
    import numpy as np
except ImportError:
    np = None
try:
    import pandas as pd
except ImportError:
    pd = None
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Union

from data_processing import load_config, preprocess
from explainability import load_artifacts, get_top_contributors
from mitigation_engine import recommend_actions

app = FastAPI()

# Load configuration once at startup
config = load_config("config.yaml")

# Load model artifacts (model, encoder, scaler, config) – path relative to this file
model_path = "models/best_model.joblib"
model, encoder, scaler, cfg = load_artifacts(model_path)

# Determine feature order after preprocessing (same as in training)
# Load a small sample from the training data to get feature names
try:
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "..", "shrimp_disease_detection_dataset_professional.csv")
    df_sample = pd.read_csv(csv_path)
    df_sample = df_sample.dropna()
    # Remove target column if it exists
    if "disease_risk_level" in df_sample.columns:
        df_sample = df_sample.drop(columns=["disease_risk_level"])
    df_sample = df_sample.head(1)
    X_sample, _, _, _ = preprocess(df_sample, cfg, training=False, encoder=encoder, scaler=scaler)
    feature_names = list(X_sample.columns)
    print(f"Successfully loaded {len(feature_names)} feature names from sample")
except Exception as e:
    # Fallback: use generic feature names if sampling fails
    print(f"Warning: Could not load feature names from sample: {e}")
    feature_names = []

class PredictRequest(BaseModel):
    pond_data: Dict[str, Union[str, float, int]]

@app.post("/predict")
def predict(request: PredictRequest):
    # Convert incoming dict to DataFrame
    df = pd.DataFrame([request.pond_data])
    # Ensure required columns exist
    missing = set(config.get("required_columns", [])) - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
    # Preprocess using same pipeline as training (pass saved encoder and scaler)
    X_processed, _, _, _ = preprocess(df, cfg, training=False, encoder=encoder, scaler=scaler)
    # Predict probabilities
    probs = model.predict_proba(X_processed)
    pred_idx = np.argmax(probs, axis=1)[0]
    disease_risk = model.classes_[pred_idx]
    confidence = float(probs[0, pred_idx])
    # Get top contributing features (local explanation)
    sample = X_processed.iloc[0].values.reshape(1, -1)
    top_factors = get_top_contributors(sample, model, feature_names, top_n=3)
    # Generate mitigation actions
    actions = recommend_actions(disease_risk, top_factors, cfg)
    # Generate natural‑language explanation via Groq
    from groq_client import generate_natural_language
    try:
        explanation = generate_natural_language(disease_risk, top_factors, actions)
    except Exception as e:
        # Fallback if Groq API fails (e.g., no API key configured)
        explanation = f"Disease Risk Level: {disease_risk}. Key Risk Factors: {', '.join(top_factors)}. Recommended Actions: {', '.join(actions)}. Note: Extended explanation requires Groq API configuration. Error: {str(e)}"
    return {
        "disease_risk": disease_risk,
        "confidence": confidence,
        "key_risk_factors": top_factors,
        "recommended_actions": actions,
        "explanation": explanation,
    }
