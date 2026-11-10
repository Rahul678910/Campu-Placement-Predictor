# PRD: Campus Placement Predictor

## Project Overview
- **Project Name:** Campus Placement Predictor (AI/ML-Based Predictive Analytics)
- **Organization / Internship:** Lazyskool (YuvaBot Pvt. Ltd.)
- **Duration:** 3 Months
- **Development Tool:** Google Antigravity

## Objective
Develop an end-to-end artificial intelligence and machine learning predictive system to forecast student campus placement outcomes, estimate prospective compensation packages, uncover recruitment correlation drivers, and deliver personalized career recommendations through an interactive web application.

---

## Task Backlog

- [x] **Task 1: Realistic Student Placement Dataset Generation Engine (`src/data_generator.py`)**
  - Synthesize a statistically sound, realistic cohort of 1,500+ student profiles.
  - Realistic feature distributions: Secondary (10th), Higher Secondary (12th), Undergrad CGPA/percentage, Stream/Discipline, Aptitude scores, Coding skill assessments, Internships, Projects, Certifications, Soft skills, Attendance, and Backlogs history.
  - Ground truth generation with realistic domain noise for placement probability and salary brackets.
  - Export raw dataset to `data/raw/student_placement_data.csv`.

- [x] **Task 2: Data Preprocessing, Cleaning & Validation Pipeline (`src/data_processing.py`)**
  - Implement Pydantic schema validation for incoming records.
  - Data cleaning: handle missing values, handle categorical consistency, detect statistical anomalies.
  - Stratified train/test splitting (80/20) and clean persistence to `data/processed/`.

- [x] **Task 3: Feature Engineering & Transformation Pipeline (`src/feature_engineering.py`)**
  - Construct domain-specific composite metrics (Academic Consistency Index, Technical Quotient, Readiness Score, Experience Weight).
  - Leakage-free scikit-learn compatible preprocessing transformer (scaling, one-hot encoding).
  - Feature selection and transformer serialization to `models/preprocessor.joblib`.

- [x] **Task 4: Exploratory Data Analysis (EDA) & Correlation Analytics (`reports/eda_insights.md` & `notebooks/`)**
  - Deep-dive distribution analysis and statistical correlation matrices.
  - Visualization artifacts: Correlation Heatmap, Academic vs Placement Trends, Coding vs Aptitude Scatter, Placement Rate by Discipline.
  - Automated report generation in `reports/figures/` and `reports/eda_insights.md`.
  - Comprehensive Jupyter notebook in `notebooks/eda_placement_analysis.ipynb`.

- [x] **Task 5: Model Training, Cross-Validation & Hyperparameter Tuning (`src/train_model.py`)**
  - Train multiple supervised classification algorithms:
    - Logistic Regression (Baseline)
    - Random Forest Classifier
    - Gradient Boosting / XGBoost Classifier
  - Train a companion Salary Regressor (Random Forest / Ridge) for placed candidates.
  - Stratified 5-fold cross-validation, hyperparameter tuning, model persistence to `models/`.

- [x] **Task 6: Model Evaluation & SHAP Interpretability (`src/evaluate_model.py`)**
  - Compute Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
  - Generate feature importance and SHAP explanations (TreeExplainer).
  - Save evaluation summary to `reports/model_evaluation.md` and charts to `reports/figures/`.

- [x] **Task 7: Interactive Streamlit Dashboard (`src/app.py`)**
  - Multi-tab application:
    1. **Executive Overview & KPIs**: Historical placement metrics, key benchmarks.
    2. **Student Profiler & Analytics**: Interactive exploratory charts and cohort filters.
    3. **Live Placement Predictor**: Real-time form for single student evaluation with probability gauge, predicted salary range, and SHAP-driven key factor explanations.
    4. **Career Guidance & Recommendations**: Actionable AI recommendations for improving placement chances (certifications, project portfolio, aptitude coaching).
  - Launch scripts (`run.bat` / `run.sh`).

- [x] **Task 8: Comprehensive Unit Testing Suite (`tests/`)**
  - Test data generation validity and boundary constraints.
  - Test preprocessing pipeline and feature transformers.
  - Test model prediction schema, probability outputs, and salary regression.
  - Execute test suite with `pytest`.

- [x] **Task 9: Internship Project Report, Documentation & Artifacts**
  - Complete `README.md` with installation, usage, and architecture overview.
  - Official Internship Report: `reports/internship_final_report.md` detailing methodology, EDA findings, model benchmarks, and business recommendations for Lazyskool / YuvaBot Pvt. Ltd.
