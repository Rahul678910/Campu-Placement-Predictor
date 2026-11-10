"""
Unit tests for feature engineering pipeline.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import (
    DomainFeatureExtractor,
    build_preprocessor,
    fit_and_save_pipeline,
    get_feature_names,
)


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame(
        [
            {
                "student_id": "STU-001",
                "gender": "Male",
                "ssc_percentage": 75.0,
                "ssc_board": "Central",
                "hsc_percentage": 70.0,
                "hsc_board": "Central",
                "hsc_stream": "Science",
                "degree_percentage": 72.0,
                "degree_field": "Computer Science",
                "internships_count": 2,
                "projects_count": 3,
                "work_experience": "No",
                "certifications_count": 1,
                "aptitude_test_score": 70.0,
                "coding_test_score": 80.0,
                "soft_skills_score": 7.0,
                "attendance_percentage": 85.0,
                "backlogs_history": 0,
                "placement_status": "Placed",
                "company_tier": "Tier 1 (Product/MNC)",
                "salary_offered": 10.5,
            },
            {
                "student_id": "STU-002",
                "gender": "Female",
                "ssc_percentage": 60.0,
                "ssc_board": "State",
                "hsc_percentage": 58.0,
                "hsc_board": "State",
                "hsc_stream": "Arts",
                "degree_percentage": 62.0,
                "degree_field": "Other",
                "internships_count": 0,
                "projects_count": 1,
                "work_experience": "No",
                "certifications_count": 0,
                "aptitude_test_score": 50.0,
                "coding_test_score": 40.0,
                "soft_skills_score": 5.5,
                "attendance_percentage": 70.0,
                "backlogs_history": 2,
                "placement_status": "Not Placed",
                "company_tier": "None",
                "salary_offered": 0.0,
            },
        ]
    )


def test_domain_feature_extractor(sample_raw_df):
    extractor = DomainFeatureExtractor()
    extracted = extractor.transform(sample_raw_df)

    assert "academic_consistency_index" in extracted.columns
    assert "technical_quotient" in extracted.columns
    assert "practical_experience_score" in extracted.columns
    assert "holistic_readiness_score" in extracted.columns
    assert "has_backlogs" in extracted.columns
    assert "has_internship" in extracted.columns
    assert "is_stem_degree" in extracted.columns

    # Check STEM mapping
    assert extracted.loc[0, "is_stem_degree"] == 1
    assert extracted.loc[1, "is_stem_degree"] == 0

    # Backlog penalty test
    assert extracted.loc[0, "has_backlogs"] == 0
    assert extracted.loc[1, "has_backlogs"] == 1
    assert extracted.loc[0, "holistic_readiness_score"] > extracted.loc[1, "holistic_readiness_score"]


def test_preprocessor_pipeline_fit_transform(sample_raw_df):
    pipeline = build_preprocessor()
    X = sample_raw_df.drop(columns=["student_id", "placement_status", "company_tier", "salary_offered"])

    pipeline.fit(X)
    transformed = pipeline.transform(X)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 20

    feature_names = get_feature_names(pipeline)
    assert len(feature_names) == transformed.shape[1]
