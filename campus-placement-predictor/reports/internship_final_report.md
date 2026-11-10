# Campus Placement Predictor: AI/ML-Based Predictive Analytics Project
## 3-Month Internship Project Final Report
**Organization:** Lazyskool (YuvaBot Pvt. Ltd.)  
**Development Tool:** Google Antigravity  
**Author:** RAM DAS MURMU  
**Domain:** Artificial Intelligence & Machine Learning  

---

### Executive Summary
Campus recruitment is a cornerstone of higher education outcomes, directly impacting student careers and institutional prestige. This project develops an end-to-end Machine Learning Predictive System designed to forecast campus placement outcomes, quantify the predictive weights of student attributes, estimate salary package brackets, and deliver actionable career remediation pathways.

The system incorporates data preprocessing, exploratory data analysis, domain-driven feature engineering, multi-model supervised classification (Logistic Regression, Random Forest, Gradient Boosting, XGBoost), compensation regression modeling, and a real-time Streamlit interactive web application. The champion classifier achieved **88.33% accuracy** and an **ROC-AUC of 0.9627** on holdout test data.

---

### 1. Project Background & Objective
#### 1.1 Organizational Context
Developed during a 3-month AI/ML internship at **Lazyskool (YuvaBot Pvt. Ltd.)**, using **Google Antigravity** as the primary agentic development environment.

#### 1.2 Problem Statement
Traditional campus placement preparation relies on generalized advice without quantitative assessments of an individual's readiness. Students often struggle to identify whether their primary barrier is academic consistency, lack of practical coding proficiency, insufficient internship exposure, or historical backlogs.

#### 1.3 Key Objectives
1. **Data Ingestion & Cleaning:** Develop a standardized student schema with validation rules and missing-value imputation.
2. **Exploratory Analytics (EDA):** Identify empirical correlations between academic track records, standardized test metrics, and placement success.
3. **Feature Engineering:** Formulate composite indices (Academic Consistency Index, Technical Quotient, Holistic Readiness Score) that eliminate leakage.
4. **Supervised Predictive Modeling:** Benchmark multiple classification algorithms via 5-Fold Stratified Cross-Validation and train a secondary salary estimation regressor.
5. **Interactive Deployment:** Deploy a production-ready dashboard enabling students and placement officers to simulate profile outcomes and receive actionable guidance.

---

### 2. Dataset Schema & Features
The dataset consists of **1,800 student records** encompassing four operational dimensions:

| Category | Features | Description |
|:---|:---|:---|
| **Academic Background** | `ssc_percentage`, `ssc_board`, `hsc_percentage`, `hsc_board`, `hsc_stream`, `degree_percentage`, `degree_field` | Standardized historical academic performance across 10th, 12th, and undergraduate degree. |
| **Technical & Assessment** | `aptitude_test_score`, `coding_test_score`, `soft_skills_score` | Standardized cognitive aptitude percentile, algorithmic coding evaluation, and communication rating. |
| **Experience & Activities** | `internships_count`, `projects_count`, `certifications_count`, `work_experience` | Practical software development experience and verified credentials. |
| **Discipline & Behavioral** | `attendance_percentage`, `backlogs_history` | Academic consistency and historical backlogs. |
| **Recruitment Outcomes** | `placement_status`, `company_tier`, `salary_offered` | Ground truth placement status, company bracket (Tier 1/2/3), and compensation in INR LPA. |

---

### 3. Exploratory Data Analysis & Empirical Insights
#### 3.1 Key Statistical Findings
- **Overall Placement Rate:** 69.8% across the cohort.
- **Top Positive Correlate:** Algorithmic coding assessment score (`+0.62` correlation with selection).
- **The Backlog Threshold:** Students with 0 backlogs maintained an **81.4%** placement conversion rate. Possessing 1 backlog dropped placement rates to **48.2%**, while 2 or more backlogs caused an abrupt decline to **< 16%** due to corporate filtering thresholds.
- **Discipline Breakdown:** Computing branches (Computer Science, Information Technology) demonstrated the highest placement conversions (**78.2%** and **75.4%** respectively).

#### 3.2 Visualizations Generated
- `reports/figures/correlation_heatmap.png`: Inter-feature correlation matrix.
- `reports/figures/coding_vs_aptitude_scatter.png`: Dual-assessment scatter identifying cutoff boundaries.
- `reports/figures/degree_vs_placement.png`: Discipline-wise conversion rates.
- `reports/figures/backlogs_impact.png`: Attrition rate across backlog tiers.
- `reports/figures/salary_distribution_by_tier.png`: Compensation ranges across corporate tiers.

---

### 4. Feature Engineering & Preprocessing
To improve discriminative capability without introducing leakage:
1. **Academic Consistency Index ($ACI$):**
   $$ACI = \frac{\text{SSC} + \text{HSC} + 2 \times \text{Degree}}{4}$$
2. **Technical Quotient ($TQ$):**
   $$TQ = 0.60 \times \text{Coding} + 0.40 \times \text{Aptitude}$$
3. **Holistic Readiness Score ($HRS$):**
   $$HRS = 0.35(TQ) + 0.25(ACI) + 0.15(\text{SoftSkills} \times 10) + 0.15(\text{ExpScore}) + 0.10(\text{Att}) - \text{BacklogPenalty}$$
4. **ColumnTransformer Pipeline:**
   - Numerical attributes: Z-score standardized with `StandardScaler()`.
   - Categorical attributes: Encoded via `OneHotEncoder(drop='first', handle_unknown='ignore')`.

---

### 5. Model Architecture & Benchmarking
#### 5.1 Stratified 5-Fold Cross-Validation Comparison
Four candidate architectures were trained and validated on the 80% training set (1,440 instances):

| Algorithm | CV ROC-AUC | CV Accuracy | CV F1-Score | CV Precision | CV Recall |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Selected)** | **0.9562** | **88.33%** | **0.9135** | **90.84%** | **91.88%** |
| **XGBoost Classifier** | 0.9431 | 86.46% | 0.9037 | 89.87% | 90.90% |
| **Random Forest Classifier** | 0.9426 | 87.22% | 0.9074 | 88.62% | 93.00% |
| **Gradient Boosting** | 0.9403 | 86.74% | 0.9056 | 88.75% | 92.50% |

#### 5.2 Holdout Test Evaluation (360 Unseen Samples)
- **Accuracy:** 88.33%
- **ROC-AUC:** 0.9627
- **Precision:** 92.65%
- **Recall:** 90.44%
- **F1-Score:** 0.9153
- **Salary Regressor MAE:** ₹1.41 LPA ($R^2 = 0.555$)

---

### 6. Interactive Web Application
Built with **Streamlit** and **Plotly**, the application features:
1. **Live Predictor:** Real-time form evaluating student profiles with a dynamic Plotly gauge, confidence tier, and projected salary package.
2. **Cohort Analytics:** Interactive exploration of distributions, correlations, and tier breakdowns.
3. **Career Guidance AI:** Personalized recommendation engine categorizing interventions into Coding/DSA, Internships, Academic Defense, and Soft Skills.
4. **Model Benchmarks:** Transparent display of CV scores, holdout metrics, and ROC curves.

---

### 7. Recommendations for Educational Institutions
1. **Early Diagnostic Testing:** Administer the predictor at the beginning of Year 3 to identify at-risk students before recruitment drives commence.
2. **Mandatory Backlog Clearance Campaigns:** Prioritize summer remedial terms to clear backlogs before final-year eligibility lists are generated.
3. **Coding Bootcamps:** Integrate hands-on problem solving into core curricula to improve coding test scores.

---

### 8. Conclusion
The Campus Placement Predictor delivers a robust, explainable, and accessible machine learning solution for student recruitment analytics. Developed during the 3-month internship at **Lazyskool (YuvaBot Pvt. Ltd.)** using **Google Antigravity**, this project demonstrates end-to-end data science maturity from raw data synthesis to production-ready UI deployment.
