import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import yaml


def load_config(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    return df


def preprocess(df: pd.DataFrame, config: dict, training: bool = True, encoder=None, scaler=None):
    # Drop rows with missing values
    df = df.dropna()
    
    # Separate target only if it exists (training mode)
    if "disease_risk_level" in df.columns and training:
        y = df["disease_risk_level"]
        X = df.drop(columns=["disease_risk_level"])
    else:
        # Prediction mode - no target column
        y = None
        X = df.copy()

    # Columns to exclude from encoding/scaling (metadata, not features)
    exclude_cols = ["pond_id", "suspected_condition", "treatment_applied", "treatment_intensity_class"]
    X = X.drop(columns=[c for c in exclude_cols if c in X.columns])
    
    # Identify categorical and numeric columns
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Start with numeric columns as-is
    X_numeric = X[num_cols].copy() if num_cols else pd.DataFrame(index=X.index)

    # Encode categoricals
    if encoder is None:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    
    if cat_cols:
        if training:
            X_cat_encoded = encoder.fit_transform(X[cat_cols])
        else:
            X_cat_encoded = encoder.transform(X[cat_cols])
        # Create a dataframe with encoded values
        X_cat_df = pd.DataFrame(X_cat_encoded, columns=cat_cols, index=X.index, dtype=float)
        # Concatenate
        X = pd.concat([X_numeric, X_cat_df], axis=1)
    else:
        X = X_numeric

    # Ensure all columns are numeric and fill any NaN with 0
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    # Drop any remaining rows with NaN
    X = X.dropna()
    if y is not None:
        y = y[X.index]

    # Scale numeric features
    if scaler is None:
        scaler = StandardScaler()
    
    all_cols = X.columns.tolist()
    if all_cols:
        if training:
            X_scaled = scaler.fit_transform(X[all_cols])
        else:
            X_scaled = scaler.transform(X[all_cols])
        X = pd.DataFrame(X_scaled, columns=all_cols, index=X.index)

    # Simple outlier capping using IQR
    for col in num_cols:
        if col in X.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            X[col] = np.clip(X[col], lower, upper)

    return X, y, encoder, scaler

    # Simple outlier capping using IQR
    for col in num_cols:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        X[col] = np.clip(X[col], lower, upper)

    return X, y, encoder, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
