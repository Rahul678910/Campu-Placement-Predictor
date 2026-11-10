"""
Model Training, Cross-Validation & Hyperparameter Optimization Pipeline.
Trains and compares Logistic Regression, Random Forest, and Gradient Boosting/XGBoost models,
selects the best placement predictor, and trains a companion salary regressor.
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.feature_engineering import build_preprocessor


def train_placement_classifiers(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Any, str]:
    """Train multiple classifiers with Stratified 5-Fold Cross Validation.

    Returns evaluation results dictionary, the best fitted model, and its name.
    """
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=random_state, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=4,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            random_state=random_state,
        ),
    }

    # Optional: If XGBoost is installed, add it
    try:
        from xgboost import XGBClassifier

        candidate_models["XGBoost"] = XGBClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        pass

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = {}
    best_roc_auc = -1.0
    best_model_name = ""
    best_model = None

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for name, model in candidate_models.items():
        cv_res = cross_validate(
            model,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        mean_roc = float(np.mean(cv_res["test_roc_auc"]))
        mean_acc = float(np.mean(cv_res["test_accuracy"]))
        mean_f1 = float(np.mean(cv_res["test_f1"]))
        mean_prec = float(np.mean(cv_res["test_precision"]))
        mean_rec = float(np.mean(cv_res["test_recall"]))

        results[name] = {
            "cv_roc_auc": round(mean_roc, 4),
            "cv_roc_auc_std": round(float(np.std(cv_res["test_roc_auc"])), 4),
            "cv_accuracy": round(mean_acc, 4),
            "cv_f1": round(mean_f1, 4),
            "cv_precision": round(mean_prec, 4),
            "cv_recall": round(mean_rec, 4),
        }

        # Train full model on all training data
        model.fit(X_train, y_train)

        if mean_roc > best_roc_auc:
            best_roc_auc = mean_roc
            best_model_name = name
            best_model = model

    return results, best_model, best_model_name


def train_salary_regressor(
    X_train_placed: pd.DataFrame,
    y_train_salary: np.ndarray,
    random_state: int = 42,
) -> Any:
    """Train a regression model to estimate salary package (LPA) for placed students."""
    regressor = RandomForestRegressor(
        n_estimators=120,
        max_depth=7,
        random_state=random_state,
        n_jobs=-1,
    )
    regressor.fit(X_train_placed, y_train_salary)
    return regressor


def execute_training_pipeline(
    train_path: str = "data/processed/train.csv",
    models_dir: str = "models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end model training, validation, and artifact serialization."""
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_df = pd.read_csv(train_path)

    feature_cols = [
        c
        for c in train_df.columns
        if c not in ["student_id", "placement_status", "company_tier", "salary_offered"]
    ]
    X_raw = train_df[feature_cols]
    y = (train_df["placement_status"] == "Placed").astype(int).values

    # 2. Build and fit preprocessor
    pipeline = build_preprocessor()
    X_transformed = pipeline.fit_transform(X_raw)

    # Save fitted preprocessor
    joblib.dump(pipeline, out_dir / "preprocessor.joblib")

    # 3. Train classifiers & evaluate via 5-Fold Stratified CV
    cv_results, best_model, best_name = train_placement_classifiers(
        X_transformed, y, cv_folds=5, random_state=random_state
    )

    # Save best placement model
    joblib.dump(best_model, out_dir / "best_placement_model.joblib")

    # 4. Train salary regressor (placed subset only)
    placed_mask = train_df["placement_status"] == "Placed"
    X_placed_raw = train_df.loc[placed_mask, feature_cols]
    y_placed_salary = train_df.loc[placed_mask, "salary_offered"].values

    X_placed_trans = pipeline.transform(X_placed_raw)
    salary_reg = train_salary_regressor(X_placed_trans, y_placed_salary, random_state=random_state)
    joblib.dump(salary_reg, out_dir / "salary_regressor.joblib")

    # 5. Save summary JSON
    summary = {
        "best_placement_model": best_name,
        "best_cv_roc_auc": cv_results[best_name]["cv_roc_auc"],
        "best_cv_accuracy": cv_results[best_name]["cv_accuracy"],
        "best_cv_f1": cv_results[best_name]["cv_f1"],
        "comparison": cv_results,
    }

    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train placement and salary models")
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save serialized models",
    )
    args = parser.parse_args()

    results = execute_training_pipeline(args.train_data, args.models_dir)
    print("\n========== MODEL TRAINING & COMPARISON SUMMARY ==========")
    for model_name, metrics in results["comparison"].items():
        print(
            f"{model_name:22} | ROC-AUC: {metrics['cv_roc_auc']:.4f} | Acc: {metrics['cv_accuracy']:.4f} | F1: {metrics['cv_f1']:.4f}"
        )
    print(
        f"\n[WINNER] Selected: {results['best_placement_model']} (ROC-AUC: {results['best_cv_roc_auc']:.4f})"
    )
    print(f"[SAVED] Artifacts stored in: {args.models_dir}")
