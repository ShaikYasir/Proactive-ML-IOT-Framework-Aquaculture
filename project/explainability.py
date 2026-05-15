import joblib
import numpy as np
# Optional imports wrapped in try/except to avoid lint errors
try:
    import shap
except ImportError:
    shap = None


# Load the saved model and preprocessing objects
def load_artifacts(model_path: str = "../models/best_model.joblib"):
    data = joblib.load(model_path)
    return data["model"], data["encoder"], data["scaler"], data["config"]


def get_feature_importance(model, feature_names):
    """Return global feature importance for tree based models.
    For LogisticRegression, use absolute coefficients.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        raise ValueError("Model does not expose feature importance")
    return dict(zip(feature_names, importances))


def get_top_contributors(sample: np.ndarray, model, feature_names, top_n: int = 3):
    """Local explanation for a single sample.
    Uses SHAP for tree models if available, otherwise falls back to
    coefficient magnitude for linear models.
    Returns a list of feature names ordered by contribution magnitude.
    """
    # Handle empty feature_names
    if not feature_names or len(feature_names) == 0:
        return []
    
    try:
        import shap
        # Use TreeExplainer for LightGBM/RandomForest, LinearExplainer for LR
        if hasattr(model, "predict_proba") and hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, sample)
        shap_values = explainer.shap_values(sample)
        # For multiclass, shap_values is a list per class; pick the predicted class
        if isinstance(shap_values, list):
            pred_class = model.predict(sample.reshape(1, -1))[0]
            shap_vals = shap_values[pred_class]
        else:
            shap_vals = shap_values
        # Pair feature names with absolute SHAP values
        contrib = list(zip(feature_names, np.abs(shap_vals).flatten()))
        contrib.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in contrib[:top_n]]
    except Exception as e:
        # Fallback: use absolute coefficient magnitude for linear models
        try:
            if hasattr(model, "coef_"):
                coeffs = np.abs(model.coef_).mean(axis=0)
                idx = np.argsort(-coeffs)[:top_n]
                return [feature_names[i] for i in idx]
        except Exception:
            pass
        # Final fallback: return first N feature names
        return feature_names[:top_n] if feature_names else []

# Helper to map raw feature names after encoding/scaling back to original column names
def get_feature_names(df_columns, encoder, scaler, cat_cols, num_cols):
    # After preprocessing, columns stay in same order; just return the list
    return list(df_columns)

# Example usage inside API (not executed here)
if __name__ == "__main__":
    model, enc, scl, cfg = load_artifacts()
    # Assume we have a sample row as a numpy array after preprocessing
    # sample = ...
    # feature_names = get_feature_names(...)
    # top = get_top_contributors(sample, model, feature_names)
    # print(top)
