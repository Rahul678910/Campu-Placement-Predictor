"""
Data Generation Engine for Campus Placement Predictor.
Generates statistically realistic student placement datasets with realistic
covariates, academic records, aptitude metrics, behavioral attributes, and ground truth outcomes.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def generate_placement_dataset(
    n_samples: int = 1800,
    random_state: int = 42,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Generate realistic student cohort data with academic and placement records.

    Parameters
    ----------
    n_samples : int
        Number of student profiles to generate.
    random_state : int
        Seed for reproducible statistical generation.
    output_path : Path | str | None
        Optional file path to persist the generated CSV dataset.

    Returns
    -------
    pd.DataFrame
        Synthetic student placement DataFrame.
    """
    rng = np.random.default_rng(random_state)

    # 1. Demographic & Basic Identifiers
    student_ids = [f"STU-2024-{i + 1:04d}" for i in range(n_samples)]
    genders = rng.choice(["Male", "Female"], size=n_samples, p=[0.56, 0.44])

    # 2. Secondary (10th) & Higher Secondary (12th) Academics
    ssc_boards = rng.choice(["Central", "State", "ICSE"], size=n_samples, p=[0.48, 0.38, 0.14])
    ssc_percentage = np.clip(rng.normal(loc=73.5, scale=9.8, size=n_samples), 45.0, 99.0)

    # HSC features are moderately correlated with SSC
    hsc_boards = rng.choice(["Central", "State", "ISC"], size=n_samples, p=[0.46, 0.40, 0.14])
    hsc_streams = rng.choice(["Science", "Commerce", "Arts"], size=n_samples, p=[0.68, 0.25, 0.07])
    hsc_percentage = np.clip(
        0.65 * ssc_percentage + rng.normal(loc=24.0, scale=8.2, size=n_samples),
        45.0,
        99.0,
    )

    # 3. Undergraduate Academic Details
    degree_fields = rng.choice(
        [
            "Computer Science",
            "Information Technology",
            "Electronics & Comm",
            "Mechanical",
            "Commerce & Management",
            "Other",
        ],
        size=n_samples,
        p=[0.36, 0.24, 0.16, 0.10, 0.10, 0.04],
    )

    # Degree percentage / CGPA conversion (% equivalent)
    degree_percentage = np.clip(
        0.45 * hsc_percentage + rng.normal(loc=38.0, scale=7.0, size=n_samples),
        50.0,
        97.5,
    )

    # Attendance (%)
    attendance_percentage = np.clip(rng.normal(loc=82.0, scale=8.5, size=n_samples), 60.0, 99.5)

    # Historical Backlogs (0 is typical, 1-2 minor penalty, 3+ severe penalty)
    backlogs_history = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.74, 0.15, 0.06, 0.03, 0.02])

    # 4. Technical Aptitude & Practical Skills
    # Discipline-dependent coding aptitude baseline
    field_coding_boost = {
        "Computer Science": 16.0,
        "Information Technology": 14.0,
        "Electronics & Comm": 6.0,
        "Mechanical": -4.0,
        "Commerce & Management": -8.0,
        "Other": -6.0,
    }
    base_coding = np.array([field_coding_boost[f] for f in degree_fields])

    coding_test_score = np.clip(
        0.4 * degree_percentage + base_coding + rng.normal(loc=35.0, scale=14.0, size=n_samples),
        15.0,
        100.0,
    )

    aptitude_test_score = np.clip(
        0.35 * ssc_percentage + 0.35 * hsc_percentage + rng.normal(loc=18.0, scale=11.0, size=n_samples),
        25.0,
        99.0,
    )

    soft_skills_score = np.clip(
        rng.normal(loc=6.8, scale=1.4, size=n_samples),
        1.5,
        10.0,
    )

    # 5. Internships, Projects, Certifications, Work Exp
    internships_count = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.38, 0.36, 0.18, 0.06, 0.02])
    projects_count = rng.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.12, 0.32, 0.30, 0.16, 0.07, 0.03])
    certifications_count = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.28, 0.38, 0.22, 0.09, 0.03])
    work_experience = rng.choice(["No", "Yes"], size=n_samples, p=[0.78, 0.22])

    # 6. Realistic Latent Placement Propensity Score
    work_exp_num = (work_experience == "Yes").astype(float)

    # Standardize continuous factors to comparable scale
    norm_degree = (degree_percentage - 70.0) / 9.0
    norm_coding = (coding_test_score - 65.0) / 16.0
    norm_aptitude = (aptitude_test_score - 68.0) / 13.0
    norm_soft = (soft_skills_score - 6.5) / 1.5
    norm_attendance = (attendance_percentage - 80.0) / 9.0

    latent_propensity = (
        0.95 * norm_coding
        + 0.85 * norm_degree
        + 0.65 * norm_aptitude
        + 0.55 * norm_soft
        + 0.45 * internships_count
        + 0.35 * (projects_count - 2.5)
        + 0.40 * certifications_count
        + 0.50 * work_exp_num
        + 0.20 * norm_attendance
        - 1.10 * (backlogs_history > 0).astype(float)
        - 1.50 * (backlogs_history >= 2).astype(float)
        + rng.normal(loc=0.0, scale=0.75, size=n_samples)  # Realistic observational noise
    )

    # Convert to placement probability via sigmoid
    placement_prob = 1.0 / (1.0 + np.exp(-(latent_propensity - 0.25)))
    placement_status = np.where(placement_prob >= 0.50, "Placed", "Not Placed")

    # 7. Salary Package and Company Tier (if Placed)
    company_tiers = []
    salaries = []

    for i in range(n_samples):
        if placement_status[i] == "Not Placed":
            company_tiers.append("Not Applicable")
            salaries.append(0.0)
        else:
            # Placed: determine tier based on aptitude, coding score, internships & degree
            tier_score = (
                0.40 * coding_test_score[i]
                + 0.25 * aptitude_test_score[i]
                + 0.20 * degree_percentage[i]
                + 4.0 * internships_count[i]
                + 3.0 * (1 if work_experience[i] == "Yes" else 0)
                + rng.normal(0, 5.0)
            )

            if tier_score >= 78.0:
                tier = "Tier 1 (Product/MNC)"
                # 9.0 to 24.0 LPA
                base_sal = 9.0 + (tier_score - 78.0) * 0.45 + rng.normal(1.5, 1.2)
                salary = np.clip(base_sal, 9.0, 24.5)
            elif tier_score >= 62.0:
                tier = "Tier 2 (Growth/Mid-sized)"
                # 5.5 to 9.5 LPA
                base_sal = 5.5 + (tier_score - 62.0) * 0.22 + rng.normal(0.5, 0.6)
                salary = np.clip(base_sal, 5.5, 9.5)
            else:
                tier = "Tier 3 (Service/Mass)"
                # 3.2 to 5.5 LPA
                base_sal = 3.3 + (tier_score - 45.0) * 0.12 + rng.normal(0.2, 0.4)
                salary = np.clip(base_sal, 3.2, 5.5)

            company_tiers.append(tier)
            salaries.append(round(float(salary), 2))

    df = pd.DataFrame(
        {
            "student_id": student_ids,
            "gender": genders,
            "ssc_percentage": np.round(ssc_percentage, 2),
            "ssc_board": ssc_boards,
            "hsc_percentage": np.round(hsc_percentage, 2),
            "hsc_board": hsc_boards,
            "hsc_stream": hsc_streams,
            "degree_percentage": np.round(degree_percentage, 2),
            "degree_field": degree_fields,
            "internships_count": internships_count,
            "projects_count": projects_count,
            "work_experience": work_experience,
            "certifications_count": certifications_count,
            "aptitude_test_score": np.round(aptitude_test_score, 2),
            "coding_test_score": np.round(coding_test_score, 2),
            "soft_skills_score": np.round(soft_skills_score, 2),
            "attendance_percentage": np.round(attendance_percentage, 2),
            "backlogs_history": backlogs_history,
            "placement_status": placement_status,
            "company_tier": company_tiers,
            "salary_offered": salaries,
        }
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic campus placement dataset")
    parser.add_argument("--samples", type=int, default=1800, help="Number of student records")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/student_placement_data.csv",
        help="Target output CSV file path",
    )
    args = parser.parse_args()

    generated_df = generate_placement_dataset(
        n_samples=args.samples,
        random_state=args.seed,
        output_path=args.output,
    )
    placed_pct = (generated_df["placement_status"] == "Placed").mean() * 100
    print(
        f"[SUCCESS] Generated {len(generated_df)} student records with {placed_pct:.1f}% placement rate."
    )
    print(f"[SAVED] Dataset saved to: {args.output}")
