"""
Data Preprocessing, Cleaning & Validation Pipeline for Campus Placement Predictor.
Handles schema validation, missing data imputation, range sanity checks,
and stratified train/test partitioning.
"""

from pathlib import Path
from typing import Literal, Tuple
import argparse
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split


class StudentProfileSchema(BaseModel):
    """Validation schema for a single student profile."""

    student_id: str
    gender: Literal["Male", "Female", "Other"]
    ssc_percentage: float = Field(ge=35.0, le=100.0)
    ssc_board: Literal["Central", "State", "ICSE"]
    hsc_percentage: float = Field(ge=35.0, le=100.0)
    hsc_board: Literal["Central", "State", "ISC"]
    hsc_stream: Literal["Science", "Commerce", "Arts"]
    degree_percentage: float = Field(ge=35.0, le=100.0)
    degree_field: Literal[
        "Computer Science",
        "Information Technology",
        "Electronics & Comm",
        "Mechanical",
        "Commerce & Management",
        "Other",
    ]
    internships_count: int = Field(ge=0, le=10)
    projects_count: int = Field(ge=0, le=20)
    work_experience: Literal["Yes", "No"]
    certifications_count: int = Field(ge=0, le=10)
    aptitude_test_score: float = Field(ge=0.0, le=100.0)
    coding_test_score: float = Field(ge=0.0, le=100.0)
    soft_skills_score: float = Field(ge=0.0, le=10.0)
    attendance_percentage: float = Field(ge=0.0, le=100.0)
    backlogs_history: int = Field(ge=0, le=20)
    placement_status: Literal["Placed", "Not Placed"]
    company_tier: Literal[
        "Tier 1 (Product/MNC)",
        "Tier 2 (Growth/Mid-sized)",
        "Tier 3 (Service/Mass)",
        "Not Applicable",
        "None",
    ]
    salary_offered: float = Field(ge=0.0)


def clean_placement_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, standardize, and impute placement records.

    Parameters
    ----------
    df : pd.DataFrame
        Raw placement DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned and standardized DataFrame.
    """
    cleaned = df.copy()

    # Strip whitespaces from string columns
    str_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in str_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    # Ensure numerical types
    num_cols = [
        "ssc_percentage",
        "hsc_percentage",
        "degree_percentage",
        "aptitude_test_score",
        "coding_test_score",
        "soft_skills_score",
        "attendance_percentage",
        "salary_offered",
    ]
    for col in num_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    int_cols = [
        "internships_count",
        "projects_count",
        "certifications_count",
        "backlogs_history",
    ]
    for col in int_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(0).astype(int)

    # Median imputation for continuous variables if missing
    for col in num_cols:
        if col in cleaned.columns and cleaned[col].isna().any():
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)

    # Sanity clipping for percentage & score bounds
    cleaned["ssc_percentage"] = cleaned["ssc_percentage"].clip(35.0, 100.0)
    cleaned["hsc_percentage"] = cleaned["hsc_percentage"].clip(35.0, 100.0)
    cleaned["degree_percentage"] = cleaned["degree_percentage"].clip(35.0, 100.0)
    cleaned["aptitude_test_score"] = cleaned["aptitude_test_score"].clip(0.0, 100.0)
    cleaned["coding_test_score"] = cleaned["coding_test_score"].clip(0.0, 100.0)
    cleaned["soft_skills_score"] = cleaned["soft_skills_score"].clip(0.0, 10.0)
    cleaned["attendance_percentage"] = cleaned["attendance_percentage"].clip(0.0, 100.0)

    # Guarantee unplaced logic has 0 salary and 'Not Applicable' tier
    not_placed_mask = cleaned["placement_status"] == "Not Placed"
    cleaned.loc[not_placed_mask, "salary_offered"] = 0.0
    cleaned.loc[not_placed_mask, "company_tier"] = "Not Applicable"
    cleaned["company_tier"] = (
        cleaned["company_tier"].replace("None", "Not Applicable").fillna("Not Applicable")
    )

    return cleaned


def split_and_save_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    processed_dir: Path | str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform stratified split by placement_status and save datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned placement DataFrame.
    test_size : float
        Proportion of dataset to allocate to the test partition.
    random_state : int
        Seed for reproducibility.
    processed_dir : Path | str
        Destination directory for processed artifacts.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["placement_status"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save to both CSV and Parquet for flexibility
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    try:
        train_df.to_parquet(out_dir / "train.parquet", index=False)
        test_df.to_parquet(out_dir / "test.parquet", index=False)
    except Exception:
        # If pyarrow/fastparquet is not available, CSV is always present
        pass

    return train_df, test_df


def process_raw_dataset(
    raw_path: Path | str = "data/raw/student_placement_data.csv",
    processed_dir: Path | str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full execution pipeline from raw CSV to cleaned train/test splits."""
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")

    raw_df = pd.read_csv(raw_file)
    cleaned_df = clean_placement_data(raw_df)
    train_df, test_df = split_and_save_data(cleaned_df, processed_dir=processed_dir)
    return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and split campus placement dataset")
    parser.add_argument(
        "--raw",
        type=str,
        default="data/raw/student_placement_data.csv",
        help="Path to raw dataset",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory to save train and test splits",
    )
    args = parser.parse_args()

    train, test = process_raw_dataset(args.raw, args.processed_dir)
    print(
        f"[SUCCESS] Processed dataset: Train set: {train.shape[0]} rows, Test set: {test.shape[0]} rows."
    )
    print(f"[SAVED] Artifacts stored in: {args.processed_dir}")
