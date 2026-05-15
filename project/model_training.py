try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import yaml
except ImportError:
    yaml = None
try:
    import joblib
except ImportError:
    joblib = None
from pathlib import Path
try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None
try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    RandomForestClassifier = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from data_processing import load_config, load_dataset, preprocess, split_data


def train_models(csv_path: str, config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    df = load_dataset(csv_path)
    X, y, encoder, scaler = preprocess(df, cfg, training=True)
    X_train, X_val, y_train, y_val = split_data(
        X, y, test_size=cfg["training"]["test_size"], random_state=cfg["training"]["random_state"]
    )

    models = {}
    # Logistic Regression (baseline)
    lr = LogisticRegression(max_iter=200, class_weight="balanced")
    lr.fit(X_train, y_train)
    models["logistic"] = lr

    # Shallow Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=cfg["training"]["random_state"], class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    # LightGBM (fast inference)
    lgbm = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(y.unique()),
        max_depth=5,
        learning_rate=0.1,
        n_estimators=200,
        random_state=cfg["training"]["random_state"],
        class_weight="balanced",
    )
    lgbm.fit(X_train, y_train)
    models["lightgbm"] = lgbm

    # Evaluate and select best based on recall (primary metric)
    best_model_name = None
    best_recall = -1
    eval_results = {}
    for name, model in models.items():
        preds = model.predict(X_val)
        prec = precision_score(y_val, preds, average="macro", zero_division=0)
        rec = recall_score(y_val, preds, average="macro", zero_division=0)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        cm = confusion_matrix(y_val, preds)
        eval_results[name] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
        }
        # weighted recall using user weight
        weighted_rec = rec * cfg["training"].get("recall_weight", 1.0)
        if weighted_rec > best_recall:
            best_recall = weighted_rec
            best_model_name = name

    best_model = models[best_model_name]
    # Save model and preprocessing objects
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": best_model,
        "encoder": encoder,
        "scaler": scaler,
        "config": cfg,
        "eval": eval_results,
        "best_model_name": best_model_name,
    }, model_dir / "best_model.joblib")
    return best_model_name, eval_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train shrimp disease detection models")
    parser.add_argument("csv_path", help="Path to the dataset CSV file")
    parser.add_argument("--config", default="config.yaml", help="Path to config yaml")
    args = parser.parse_args()
    best, results = train_models(args.csv_path, args.config)
    print(f"Best model: {best}")
    print("Evaluation results:")
    for m, r in results.items():
        print(m, r)
