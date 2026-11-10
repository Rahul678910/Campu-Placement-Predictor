"""
Feature Engineering and Pipeline Transformer for Campus Placement Predictor.
Constructs composite indices (Academic Consistency, Technical Quotient, Experience Score)
and builds a leakage-safe scikit-learn ColumnTransformer.
"""

from pathlib import Path
from typing import List, Tuple
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DomainFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer to compute domain-specific features without leakage."""

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Academic Consistency Index (Weighted towards collegiate performance)
        ssc = df["ssc_percentage"].astype(float)
        hsc = df["hsc_percentage"].astype(float)
        deg = df["degree_percentage"].astype(float)
        df["academic_consistency_index"] = np.round((ssc + hsc + 2.0 * deg) / 4.0, 2)
        df["academic_volatility"] = np.round(np.std([ssc, hsc, deg], axis=0), 2)

        # 2. Technical Quotient (Coding + Aptitude combination)
        coding = df["coding_test_score"].astype(float)
        aptitude = df["aptitude_test_score"].astype(float)
        df["technical_quotient"] = np.round(0.60 * coding + 0.40 * aptitude, 2)

        # 3. Practical Experience Metric
        internships = df["internships_count"].astype(int)
        projects = df["projects_count"].astype(int)
        certs = df["certifications_count"].astype(int)
        work_exp = (df["work_experience"] == "Yes").astype(float)

        df["practical_experience_score"] = np.round(
            internships * 2.5 + projects * 1.0 + certs * 1.5 + work_exp * 3.0, 2
        )

        # 4. Behavioral & Preparedness Index
        soft = df["soft_skills_score"].astype(float) * 10.0  # Scale to 0-100
        attendance = df["attendance_percentage"].astype(float)
        backlogs = df["backlogs_history"].astype(int)

        # Holistic index penalizing heavy backlogs
        backlog_penalty = np.where(backlogs == 0, 0.0, np.where(backlogs == 1, 8.0, 20.0))
        df["holistic_readiness_score"] = np.round(
            0.35 * df["technical_quotient"]
            + 0.25 * df["academic_consistency_index"]
            + 0.15 * soft
            + 0.15 * df["practical_experience_score"]
            + 0.10 * attendance
            - backlog_penalty,
            2,
        )

        # 5. Discrete Flags
        df["has_backlogs"] = (backlogs > 0).astype(int)
        df["has_internship"] = (internships > 0).astype(int)
        df["is_stem_degree"] = df["degree_field"].isin(
            ["Computer Science", "Information Technology", "Electronics & Comm", "Mechanical"]
        ).astype(int)

        return df


def build_preprocessor() -> Pipeline:
    """Build a complete scikit-learn preprocessing pipeline."""
    categorical_cols = [
        "gender",
        "ssc_board",
        "hsc_board",
        "hsc_stream",
        "degree_field",
        "work_experience",
    ]

    numeric_cols = [
        "ssc_percentage",
        "hsc_percentage",
        "degree_percentage",
        "internships_count",
        "projects_count",
        "certifications_count",
        "aptitude_test_score",
        "coding_test_score",
        "soft_skills_score",
        "attendance_percentage",
        "backlogs_history",
        "academic_consistency_index",
        "academic_volatility",
        "technical_quotient",
        "practical_experience_score",
        "holistic_readiness_score",
        "has_backlogs",
        "has_internship",
        "is_stem_degree",
    ]

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    full_pipeline = Pipeline(
        steps=[
            ("domain_features", DomainFeatureExtractor()),
            ("column_transform", col_transformer),
        ]
    )

    return full_pipeline


def get_feature_names(fitted_pipeline: Pipeline) -> List[str]:
    """Retrieve output feature names from the fitted pipeline."""
    col_trans = fitted_pipeline.named_steps["column_transform"]
    num_cols = col_trans.transformers_[0][2]
    cat_cols = col_trans.transformers_[1][2]
    encoder = col_trans.transformers_[1][1]

    cat_feature_names = list(encoder.get_feature_names_out(cat_cols))
    return list(num_cols) + cat_feature_names


def fit_and_save_pipeline(
    train_path: str = "data/processed/train.csv",
    models_dir: str = "models",
) -> Tuple[Pipeline, List[str]]:
    """Fit feature pipeline on training data and serialize it."""
    train_df = pd.read_csv(train_path)

    # Exclude targets and identifiers
    drop_cols = ["student_id", "placement_status", "company_tier", "salary_offered"]
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])

    pipeline = build_preprocessor()
    pipeline.fit(X_train)

    feature_names = get_feature_names(pipeline)

    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, out_dir / "preprocessor.joblib")
    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    return pipeline, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit and save feature engineering preprocessor")
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train.csv",
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save preprocessor artifact",
    )
    args = parser.parse_args()

    pipeline, feature_names = fit_and_save_pipeline(args.train_data, args.models_dir)
    print(f"[SUCCESS] Feature pipeline fitted with {len(feature_names)} features.")
    print(f"[SAVED] Preprocessor saved to {args.models_dir}/preprocessor.joblib")
    print(f"[SAVED] Feature names saved to {args.models_dir}/feature_names.json")
