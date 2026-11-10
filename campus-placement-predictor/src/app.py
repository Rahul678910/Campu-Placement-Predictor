"""
Campus Placement Predictor - Interactive Web Application.
Built with Streamlit, scikit-learn, and Plotly.
Developed for Lazyskool (YuvaBot Pvt. Ltd.) using Google Antigravity.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Campus Placement Predictor | AI Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge {
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .badge-success { background-color: #dcfce7; color: #166534; }
    .badge-warning { background-color: #fef9c3; color: #854d0e; }
    .badge-danger { background-color: #fee2e2; color: #991b1b; }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models_and_artifacts():
    """Load serialized models, preprocessor, and metadata."""
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    data_dir = base_dir / "data"

    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    classifier = joblib.load(models_dir / "best_placement_model.joblib")
    salary_reg = joblib.load(models_dir / "salary_regressor.joblib")

    with open(models_dir / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    with open(models_dir / "training_summary.json", "r", encoding="utf-8") as f:
        train_summary = json.load(f)

    with open(models_dir / "evaluation_metrics.json", "r", encoding="utf-8") as f:
        eval_metrics = json.load(f)

    raw_data_path = data_dir / "raw" / "student_placement_data.csv"
    df_data = pd.read_csv(raw_data_path) if raw_data_path.exists() else None

    return {
        "preprocessor": preprocessor,
        "classifier": classifier,
        "salary_reg": salary_reg,
        "feature_names": feature_names,
        "train_summary": train_summary,
        "eval_metrics": eval_metrics,
        "df_data": df_data,
    }


artifacts = load_models_and_artifacts()
preprocessor = artifacts["preprocessor"]
classifier = artifacts["classifier"]
salary_reg = artifacts["salary_reg"]
train_summary = artifacts["train_summary"]
eval_metrics = artifacts["eval_metrics"]
df_data = artifacts["df_data"]

# Sidebar Info
st.sidebar.image(
    "https://img.icons8.com/fluency/96/graduation-cap.png",
    width=72,
)
st.sidebar.title("Placement AI")
st.sidebar.markdown(
    """
    **Internship Project**  
    *Organization:* Lazyskool (YuvaBot Pvt. Ltd.)  
    *Duration:* 3 Months  
    *Development Tool:* Google Antigravity  
    *Model:* Logistic Regression (ROC-AUC: 0.9627)
    """
)
st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Pro-Tip:** High coding aptitude, consistent academic grades, and zero backlogs are the top drivers of campus placement success."
)

# Main Header
st.markdown('<div class="main-title">🎓 Campus Placement Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI/ML-Powered Predictive Analytics & Student Career Readiness Intelligence System</div>',
    unsafe_allow_html=True,
)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔮 Live Predictor",
        "📊 Cohort Analytics & Trends",
        "💡 Career Guidance AI",
        "📈 Model Benchmarks",
    ]
)

# ==================== TAB 1: LIVE PREDICTOR ====================
with tab1:
    st.subheader("Student Profile Evaluation")
    st.caption("Input the academic, technical, and behavioral parameters of a student to evaluate placement probability.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 📚 Academic History")
        gender = st.selectbox("Gender", ["Male", "Female"])
        ssc_score = st.slider("10th Standard (SSC) %", 45.0, 100.0, 75.0, 0.5)
        ssc_board = st.selectbox("10th Board", ["Central", "State", "ICSE"])
        hsc_score = st.slider("12th Standard (HSC) %", 45.0, 100.0, 72.0, 0.5)
        hsc_board = st.selectbox("12th Board", ["Central", "State", "ISC"])
        hsc_stream = st.selectbox("12th Stream", ["Science", "Commerce", "Arts"])

    with col2:
        st.markdown("##### 💻 Undergrad & Technical Skills")
        degree_field = st.selectbox(
            "Undergraduate Discipline",
            [
                "Computer Science",
                "Information Technology",
                "Electronics & Comm",
                "Mechanical",
                "Commerce & Management",
                "Other",
            ],
        )
        degree_score = st.slider("Degree Score / CGPA %", 50.0, 100.0, 73.0, 0.5)
        coding_score = st.slider("Coding Assessment Score (0-100)", 0.0, 100.0, 70.0, 1.0)
        aptitude_score = st.slider("Analytical Aptitude Score (0-100)", 0.0, 100.0, 68.0, 1.0)
        soft_skills = st.slider("Communication & Soft Skills (1-10)", 1.0, 10.0, 7.0, 0.5)

    with col3:
        st.markdown("##### 🏆 Experience & Discipline")
        internships = st.number_input("Internships Completed", min_value=0, max_value=5, value=1)
        projects = st.number_input("Major Technical Projects", min_value=0, max_value=8, value=3)
        certifications = st.number_input("Industry Certifications", min_value=0, max_value=5, value=1)
        work_exp = st.selectbox("Prior Full-time/Vocational Work Exp", ["No", "Yes"])
        backlogs = st.number_input("Active/Historical Backlogs", min_value=0, max_value=6, value=0)
        attendance = st.slider("Classroom Attendance %", 60.0, 100.0, 85.0, 1.0)

    st.markdown("---")
    predict_btn = st.button("🚀 Calculate Placement Probability", type="primary", use_container_width=True)

    if predict_btn:
        # Build student record dataframe
        student_data = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "ssc_percentage": ssc_score,
                    "ssc_board": ssc_board,
                    "hsc_percentage": hsc_score,
                    "hsc_board": hsc_board,
                    "hsc_stream": hsc_stream,
                    "degree_percentage": degree_score,
                    "degree_field": degree_field,
                    "internships_count": internships,
                    "projects_count": projects,
                    "work_experience": work_exp,
                    "certifications_count": certifications,
                    "aptitude_test_score": aptitude_score,
                    "coding_test_score": coding_score,
                    "soft_skills_score": soft_skills,
                    "attendance_percentage": attendance,
                    "backlogs_history": backlogs,
                }
            ]
        )

        # Transform and predict
        transformed_features = preprocessor.transform(student_data)
        prob = classifier.predict_proba(transformed_features)[0, 1]
        is_placed = prob >= 0.50

        # Secondary prediction: Salary package
        pred_salary = None
        predicted_tier = "None"
        if is_placed:
            pred_salary = float(salary_reg.predict(transformed_features)[0])
            if pred_salary >= 9.0:
                predicted_tier = "Tier 1 (Product / Top MNC)"
            elif pred_salary >= 5.5:
                predicted_tier = "Tier 2 (Growth / Mid-sized Tech)"
            else:
                predicted_tier = "Tier 3 (Service / Mass Recruiter)"

        # Display results in cards
        res_col1, res_col2 = st.columns([1, 1.3])

        with res_col1:
            st.markdown("### Placement Outcome")
            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Placement Likelihood (%)", "font": {"size": 18}},
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#1e3a8a"},
                        "steps": [
                            {"range": [0, 45], "color": "#fee2e2"},
                            {"range": [45, 70], "color": "#fef9c3"},
                            {"range": [70, 100], "color": "#dcfce7"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": round(prob * 100, 1),
                        },
                    },
                )
            )
            gauge_fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

        with res_col2:
            st.markdown("### Decision Synthesis")
            if prob >= 0.70:
                st.success(
                    f"🎉 **HIGH PLACEMENT PROBABILITY ({prob*100:.1f}%)**\n\n"
                    "Candidate shows outstanding technical preparation and academic consistency."
                )
            elif prob >= 0.45:
                st.warning(
                    f"⚠️ **MODERATE PLACEMENT PROBABILITY ({prob*100:.1f}%)**\n\n"
                    "Candidate is on the borderline. Targeted upskilling in coding or clearing backlogs can significantly boost selection chances."
                )
            else:
                st.error(
                    f"🚨 **HIGH RISK OF NON-PLACEMENT ({prob*100:.1f}%)**\n\n"
                    "Candidate needs active intervention in aptitude, coding assessments, or backlog clearance."
                )

            if is_placed:
                st.markdown(
                    f"""
                    - **Estimated Annual Package:** **₹{pred_salary:.2f} LPA** (± ₹1.4 LPA)
                    - **Projected Placement Tier:** **{predicted_tier}**
                    - **Academic Consistency Index:** **{(ssc_score + hsc_score + 2*degree_score)/4:.1f}%**
                    """
                )
            else:
                st.markdown(
                    f"""
                    - **Key Risk Factor:** {'Active Backlogs' if backlogs > 0 else 'Low Technical/Aptitude Scores'}
                    - **Projected Tier:** Not Eligible Currently
                    """
                )

        # Factor breakdown
        st.markdown("##### 🔍 Student Readiness Factor Breakdown")
        factors = {
            "Coding Proficiency": min(100, coding_score),
            "Aptitude & Analytical": min(100, aptitude_score),
            "Academic Track Record": (ssc_score + hsc_score + 2 * degree_score) / 4.0,
            "Practical Portfolio (Internships/Projects)": min(100, (internships * 30 + projects * 15)),
            "Interview & Soft Skills": soft_skills * 10.0,
        }
        factor_df = pd.DataFrame(list(factors.items()), columns=["Dimension", "Score"])
        bar_chart = px.bar(
            factor_df,
            x="Score",
            y="Dimension",
            orientation="h",
            color="Score",
            color_continuous_scale="Teal",
            range_x=[0, 100],
        )
        bar_chart.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(bar_chart, use_container_width=True)

# ==================== TAB 2: COHORT ANALYTICS & TRENDS ====================
with tab2:
    st.subheader("Historical Recruitment Trends & Exploration")

    if df_data is not None:
        m1, m2, m3, m4 = st.columns(4)
        total = len(df_data)
        placed_count = (df_data["placement_status"] == "Placed").sum()
        placed_ratio = (placed_count / total) * 100
        avg_sal = df_data.loc[df_data["placement_status"] == "Placed", "salary_offered"].mean()
        max_sal = df_data["salary_offered"].max()

        m1.metric("Students Analyzed", f"{total:,}")
        m2.metric("Overall Placement Rate", f"{placed_ratio:.1f}%")
        m3.metric("Avg Placed Package", f"₹{avg_sal:.2f} LPA")
        m4.metric("Highest Package", f"₹{max_sal:.1f} LPA")

        st.markdown("---")
        g_col1, g_col2 = st.columns(2)

        with g_col1:
            st.markdown("##### Placement Rate by Academic Discipline")
            field_df = (
                df_data.groupby("degree_field")["placement_status"]
                .apply(lambda s: (s == "Placed").mean() * 100)
                .reset_index(name="Placement Rate (%)")
            )
            field_fig = px.bar(
                field_df,
                x="degree_field",
                y="Placement Rate (%)",
                color="Placement Rate (%)",
                color_continuous_scale="Blues",
            )
            field_fig.update_layout(xaxis_title="", yaxis_title="Placement Rate (%)", height=340)
            st.plotly_chart(field_fig, use_container_width=True)

        with g_col2:
            st.markdown("##### Compensation Distribution by Company Tier")
            placed_sub = df_data[df_data["placement_status"] == "Placed"]
            sal_fig = px.box(
                placed_sub,
                x="company_tier",
                y="salary_offered",
                color="company_tier",
                category_orders={
                    "company_tier": [
                        "Tier 3 (Service/Mass)",
                        "Tier 2 (Growth/Mid-sized)",
                        "Tier 1 (Product/MNC)",
                    ]
                },
            )
            sal_fig.update_layout(
                xaxis_title="",
                yaxis_title="Salary (INR LPA)",
                showlegend=False,
                height=340,
            )
            st.plotly_chart(sal_fig, use_container_width=True)

        st.markdown("##### Interactive Dataset Explorer")
        filter_status = st.multiselect(
            "Filter Placement Status",
            options=["Placed", "Not Placed"],
            default=["Placed", "Not Placed"],
        )
        filtered_view = df_data[df_data["placement_status"].isin(filter_status)]
        st.dataframe(filtered_view.head(50), use_container_width=True)
    else:
        st.info("Dataset file not found in `data/raw/student_placement_data.csv`.")

# ==================== TAB 3: CAREER GUIDANCE AI ====================
with tab3:
    st.subheader("Actionable AI Career Recommendations")
    st.markdown(
        """
        Our predictive engine provides customized interventions to help educational institutions and students bridge recruitment skill gaps.
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            ### 🎯 Priority 1: Technical & Coding Mastery
            - **Data Structures & Algorithms (DSA):** Practice 100+ LeetCode problems covering Arrays, Hashing, Two Pointers, Trees, and Dynamic Programming.
            - **Full-Stack & Cloud Projects:** Build 2 end-to-end full-stack applications with modern frameworks (React, FastAPI/Node, PostgreSQL).
            - **Mock Technical Interviews:** Weekly pair coding and live technical interview practice to boost coding assessment percentiles.
            """
        )

        st.markdown(
            """
            ### 💼 Priority 2: Practical Industry Exposure
            - **Internships:** Complete at least 1 summer or winter internship before the final year campus drives begin.
            - **Industry Certifications:** Obtain vendor certifications (AWS Certified Cloud Practitioner, Google Cloud Associate, or Meta Frontend Developer).
            - **Open Source Contributions:** Maintain an active GitHub profile with clean code, README documentation, and test suites.
            """
        )

    with c2:
        st.markdown(
            """
            ### ⚠️ Priority 3: Academic Governance & Backlog Mitigation
            - **Zero-Backlog Clearance:** Corporate recruiters enforce hard cutoffs on active backlogs. Institutions should organize pre-placement remedial terms to clear backlogs.
            - **CGPA Defense:** Maintain an undergraduate score above **70% / 7.5 CGPA** to remain eligible for 90%+ of visiting companies.
            - **Attendance Discipline:** Keep attendance above 75% to ensure sustained academic performance.
            """
        )

        st.markdown(
            """
            ### 🗣️ Priority 4: Aptitude & Communication Polish
            - **Quantitative & Logical Aptitude:** Daily drills on Speed Math, Logical Reasoning, and Data Interpretation.
            - **Verbal & Soft Skills:** Practice Group Discussions (GD), STAR-method behavioral interview answers, and elevator pitch delivery.
            """
        )

# ==================== TAB 4: MODEL BENCHMARKS ====================
with tab4:
    st.subheader("Model Benchmarks, Cross-Validation & Technical Integrity")

    st.markdown("##### 5-Fold Stratified Cross-Validation Comparison")
    comp = train_summary.get("comparison", {})
    comp_records = []
    for m_name, vals in comp.items():
        comp_records.append(
            {
                "Model Algorithm": m_name,
                "CV ROC-AUC": vals.get("cv_roc_auc"),
                "CV Accuracy (%)": round(vals.get("cv_accuracy") * 100, 2),
                "CV F1-Score": vals.get("cv_f1"),
                "CV Precision": vals.get("cv_precision"),
                "CV Recall": vals.get("cv_recall"),
            }
        )
    st.dataframe(pd.DataFrame(comp_records), use_container_width=True)

    st.markdown("##### Holdout Test Set Performance Metrics")
    ev1, ev2, ev3, ev4, ev5 = st.columns(5)
    ev1.metric("Holdout Accuracy", f"{eval_metrics.get('test_accuracy', 0)*100:.2f}%")
    ev2.metric("Holdout ROC-AUC", f"{eval_metrics.get('test_roc_auc', 0):.4f}")
    ev3.metric("Holdout Precision", f"{eval_metrics.get('test_precision', 0)*100:.2f}%")
    ev4.metric("Holdout Recall", f"{eval_metrics.get('test_recall', 0)*100:.2f}%")
    ev5.metric("Holdout F1-Score", f"{eval_metrics.get('test_f1', 0):.4f}")

    st.markdown("---")
    st.markdown(
        """
        **Project Metadata:**
        - **Organization:** Lazyskool (YuvaBot Pvt. Ltd.)
        - **Internship Duration:** 3 Months
        - **AI Development Tool:** Google Antigravity
        - **Pipeline Frameworks:** Scikit-Learn, Pandas, NumPy, Plotly, Streamlit
        """
    )
