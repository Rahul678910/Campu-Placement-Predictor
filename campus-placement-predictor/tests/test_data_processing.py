"""
Unit tests for data preprocessing and validation pipeline.
"""

from pathlib import Path
import pandas as pd
import pytest
from src.data_processing import (
    StudentProfileSchema,
    clean_placement_data,
    process_raw_dataset,
    split_and_save_data,
)


def test_schema_valid_instance():
    data = {
        "student_id": "STU-2024-0001",
        "gender": "Female",
        "ssc_percentage": 78.5,
        "ssc_board": "Central",
        "hsc_percentage": 76.0,
        "hsc_board": "Central",
        "hsc_stream": "Science",
        "degree_percentage": 74.2,
        "degree_field": "Computer Science",
        "internships_count": 2,
        "projects_count": 3,
        "work_experience": "No",
        "certifications_count": 1,
        "aptitude_test_score": 75.0,
        "coding_test_score": 82.0,
        "soft_skills_score": 7.5,
        "attendance_percentage": 88.0,
        "backlogs_history": 0,
        "placement_status": "Placed",
        "company_tier": "Tier 1 (Product/MNC)",
        "salary_offered": 11.5,
    }
    profile = StudentProfileSchema(**data)
    assert profile.student_id == "STU-2024-0001"
    assert profile.placement_status == "Placed"


def test_clean_placement_data_null_imputation():
    df_with_nans = pd.DataFrame(
        [
            {
                "student_id": "STU-1",
                "gender": "Male",
                "ssc_percentage": None,
                "ssc_board": "State",
                "hsc_percentage": 65.0,
                "hsc_board": "State",
                "hsc_stream": "Science",
                "degree_percentage": 68.0,
                "degree_field": "Mechanical",
                "internships_count": 0,
                "projects_count": 2,
                "work_experience": "No",
                "certifications_count": 0,
                "aptitude_test_score": 60.0,
                "coding_test_score": 50.0,
                "soft_skills_score": 6.0,
                "attendance_percentage": 80.0,
                "backlogs_history": 1,
                "placement_status": "Not Placed",
                "company_tier": "Not Applicable",
                "salary_offered": 5.0,  # Invalid salary for Not Placed
            },
            {
                "student_id": "STU-2",
                "gender": "Female",
                "ssc_percentage": 80.0,
                "ssc_board": "Central",
                "hsc_percentage": 82.0,
                "hsc_board": "Central",
                "hsc_stream": "Commerce",
                "degree_percentage": 75.0,
                "degree_field": "Commerce & Management",
                "internships_count": 1,
                "projects_count": 3,
                "work_experience": "Yes",
                "certifications_count": 1,
                "aptitude_test_score": 70.0,
                "coding_test_score": 60.0,
                "soft_skills_score": 7.0,
                "attendance_percentage": 85.0,
                "backlogs_history": 0,
                "placement_status": "Placed",
                "company_tier": "Tier 3 (Service/Mass)",
                "salary_offered": 4.5,
            },
        ]
    )

    cleaned = clean_placement_data(df_with_nans)
    assert not cleaned["ssc_percentage"].isna().any()
    # Unplaced salary must be 0
    assert cleaned.loc[cleaned["student_id"] == "STU-1", "salary_offered"].iloc[0] == 0.0


def test_stratified_split_integrity(tmp_path):
    raw_df = pd.DataFrame(
        {
            "student_id": [f"S{i}" for i in range(100)],
            "gender": ["Male"] * 50 + ["Female"] * 50,
            "ssc_percentage": [70.0] * 100,
            "ssc_board": ["Central"] * 100,
            "hsc_percentage": [70.0] * 100,
            "hsc_board": ["Central"] * 100,
            "hsc_stream": ["Science"] * 100,
            "degree_percentage": [70.0] * 100,
            "degree_field": ["Computer Science"] * 100,
            "internships_count": [1] * 100,
            "projects_count": [2] * 100,
            "work_experience": ["No"] * 100,
            "certifications_count": [1] * 100,
            "aptitude_test_score": [65.0] * 100,
            "coding_test_score": [65.0] * 100,
            "soft_skills_score": [6.5] * 100,
            "attendance_percentage": [80.0] * 100,
            "backlogs_history": [0] * 100,
            "placement_status": ["Placed"] * 70 + ["Not Placed"] * 30,
            "company_tier": ["Tier 2 (Growth/Mid-sized)"] * 70 + ["None"] * 30,
            "salary_offered": [6.0] * 70 + [0.0] * 30,
        }
    )

    train, test = split_and_save_data(raw_df, test_size=0.20, processed_dir=tmp_path)
    assert len(train) == 80
    assert len(test) == 20
    # Stratification check (placed ratio in train and test should match 70%)
    train_rate = (train["placement_status"] == "Placed").mean()
    test_rate = (test["placement_status"] == "Placed").mean()
    assert abs(train_rate - 0.70) < 0.05
    assert abs(test_rate - 0.70) < 0.05
