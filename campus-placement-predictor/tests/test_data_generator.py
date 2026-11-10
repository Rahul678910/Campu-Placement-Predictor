"""
Unit tests for data generation engine.
"""

from pathlib import Path
import pandas as pd
import pytest
from src.data_generator import generate_placement_dataset


def test_generate_placement_dataset_shape():
    df = generate_placement_dataset(n_samples=100, random_state=123)
    assert len(df) == 100
    assert "student_id" in df.columns
    assert "placement_status" in df.columns
    assert "salary_offered" in df.columns


def test_generate_placement_dataset_ranges():
    df = generate_placement_dataset(n_samples=150, random_state=42)

    # Academic score bounds
    assert (df["ssc_percentage"] >= 45.0).all() and (df["ssc_percentage"] <= 100.0).all()
    assert (df["hsc_percentage"] >= 45.0).all() and (df["hsc_percentage"] <= 100.0).all()
    assert (df["degree_percentage"] >= 45.0).all() and (df["degree_percentage"] <= 100.0).all()

    # Scores bounds
    assert (df["aptitude_test_score"] >= 0.0).all() and (df["aptitude_test_score"] <= 100.0).all()
    assert (df["coding_test_score"] >= 0.0).all() and (df["coding_test_score"] <= 100.0).all()
    assert (df["soft_skills_score"] >= 1.0).all() and (df["soft_skills_score"] <= 10.0).all()

    # Placed vs unplaced salary consistency
    unplaced = df[df["placement_status"] == "Not Placed"]
    assert (unplaced["salary_offered"] == 0.0).all()
    assert (unplaced["company_tier"] == "Not Applicable").all()

    placed = df[df["placement_status"] == "Placed"]
    assert (placed["salary_offered"] > 0.0).all()
    assert (~placed["company_tier"].isin(["None", "Not Applicable"])).all()
