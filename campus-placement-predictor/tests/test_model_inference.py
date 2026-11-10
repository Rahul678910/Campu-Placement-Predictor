"""
Unit tests for end-to-end model inference and artifact integrity.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def artifacts():
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"

    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    classifier = joblib.load(models_dir / "best_placement_model.joblib")
    salary_reg = joblib.load(models_dir / "salary_regressor.joblib")

    with open(models_dir / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    with open(models_dir / "evaluation_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return {
        "preprocessor": preprocessor,
        "classifier": classifier,
        "salary_reg": salary_reg,
        "feature_names": feature_names,
        "metrics": metrics,
    }


def test_artifacts_exist_and_metrics_satisfy_benchmark(artifacts):
    metrics = artifacts["metrics"]
    assert metrics["test_accuracy"] >= 0.85
    assert metrics["test_roc_auc"] >= 0.90
    assert metrics["test_f1"] >= 0.85
    assert len(artifacts["feature_names"]) > 25


def test_top_tier_candidate_prediction(artifacts):
    preprocessor = artifacts["preprocessor"]
    classifier = artifacts["classifier"]
    salary_reg = artifacts["salary_reg"]

    top_student = pd.DataFrame(
        [
            {
                "gender": "Female",
                "ssc_percentage": 92.0,
                "ssc_board": "Central",
                "hsc_percentage": 90.5,
                "hsc_board": "Central",
                "hsc_stream": "Science",
                "degree_percentage": 88.0,
                "degree_field": "Computer Science",
                "internships_count": 3,
                "projects_count": 4,
                "work_experience": "Yes",
                "certifications_count": 2,
                "aptitude_test_score": 92.0,
                "coding_test_score": 95.0,
                "soft_skills_score": 9.0,
                "attendance_percentage": 95.0,
                "backlogs_history": 0,
            }
        ]
    )

    X_trans = preprocessor.transform(top_student)
    prob = classifier.predict_proba(X_trans)[0, 1]
    pred = classifier.predict(X_trans)[0]

    assert prob > 0.85
    assert pred == 1

    salary = salary_reg.predict(X_trans)[0]
    assert salary >= 8.0  # High package expectation


def test_high_risk_candidate_prediction(artifacts):
    preprocessor = artifacts["preprocessor"]
    classifier = artifacts["classifier"]

    struggling_student = pd.DataFrame(
        [
            {
                "gender": "Male",
                "ssc_percentage": 50.0,
                "ssc_board": "State",
                "hsc_percentage": 48.0,
                "hsc_board": "State",
                "hsc_stream": "Arts",
                "degree_percentage": 52.0,
                "degree_field": "Other",
                "internships_count": 0,
                "projects_count": 1,
                "work_experience": "No",
                "certifications_count": 0,
                "aptitude_test_score": 35.0,
                "coding_test_score": 25.0,
                "soft_skills_score": 3.5,
                "attendance_percentage": 62.0,
                "backlogs_history": 3,
            }
        ]
    )

    X_trans = preprocessor.transform(struggling_student)
    prob = classifier.predict_proba(X_trans)[0, 1]
    pred = classifier.predict(X_trans)[0]

    assert prob < 0.40
    assert pred == 0
