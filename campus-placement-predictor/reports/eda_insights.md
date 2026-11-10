# Campus Placement Intelligence: Exploratory Data Analysis & Statistical Report

## Executive Summary
This exploratory study investigates key drivers of campus recruitment outcomes across **1,800** undergraduate candidates. The analysis evaluates academic performance metrics, aptitude testing scores, coding proficiencies, project experience, and behavioral indicators.

- **Total Cohort Size:** 1,800 students
- **Overall Placement Rate:** 69.78% (1,256 placed)
- **Placed Candidates Average Compensation:** ₹7.56 LPA (Median: ₹7.21 LPA, Max: ₹19.39 LPA)
- **Primary Positive Driver:** `coding_test_score` (Pearson correlation: `+0.415`)
- **Primary Negative Risk Factor:** `backlogs_history` (Pearson correlation: `-0.318`)

---

## 1. Univariate & Placement Distribution Analysis

### Placement Rate by Degree Field
Discipline-wise recruitment analysis reveals higher conversion rates in computing and software-centric engineering branches:

| Academic Field / Discipline | Placement Rate (%) |
|:---|:---:|
| **Computer Science** | 75.72% |
| **Information Technology** | 74.13% |
| **Electronics & Comm** | 70.68% |
| **Other** | 62.12% |
| **Commerce & Management** | 58.24% |
| **Mechanical** | 48.52% |

![Placement Rate by Degree Field](figures/degree_vs_placement.png)

---

## 2. Statistical Correlation with Recruitment Outcomes
Pearson correlation analysis between continuous attributes and placement outcome:

| Student Attribute | Correlation with Placement | Direction |
|:---|:---:|:---:|
| `coding_test_score` | `+0.415` | Positive ↗ |
| `degree_percentage` | `+0.354` | Positive ↗ |
| `aptitude_test_score` | `+0.282` | Positive ↗ |
| `hsc_percentage` | `+0.259` | Positive ↗ |
| `ssc_percentage` | `+0.206` | Positive ↗ |
| `soft_skills_score` | `+0.160` | Positive ↗ |
| `certifications_count` | `+0.156` | Positive ↗ |
| `projects_count` | `+0.153` | Positive ↗ |
| `internships_count` | `+0.148` | Positive ↗ |
| `attendance_percentage` | `+0.092` | Positive ↗ |
| `backlogs_history` | `-0.318` | Negative ↘ |

![Correlation Heatmap](figures/correlation_heatmap.png)

### Key Takeaways:
1. **Coding & Technical Proficiency:** Demonstrates the strongest single linear correlation with selection (`+0.415`).
2. **Academic Consistency:** Degree CGPA/percentage and Higher Secondary marks exhibit solid positive associations with tier eligibility.
3. **The Backlog Penalty:** Having 1 backlog reduces probability moderately, while having 2 or more historical backlogs causes an immediate crash in campus drive eligibility due to corporate cutoff policies.

---

## 3. Interaction of Aptitude and Coding Proficiency
Recruiters typically impose dual-stage filtering: initial aptitude assessment followed by coding evaluation.

![Coding vs Aptitude Scatter](figures/coding_vs_aptitude_scatter.png)

Students scoring above **65% in both Aptitude and Coding** achieved an overwhelming **93.8% placement rate**, whereas candidates failing both benchmarks registered less than **12.4% success**.

---

## 4. Backlog Vulnerability Analysis
Academic backlogs represent the single sharpest disqualifying criterion imposed by tier-1 and tier-2 campus recruiters:

| Historical Backlogs | Placement Rate (%) |
|:---|:---:|
| **0 Backlog(s)** | 77.66% |
| **1 Backlog(s)** | 63.60% |
| **2 Backlog(s)** | 31.25% |
| **3 Backlog(s)** | 26.42% |
| **4 Backlog(s)** | 35.00% |

![Backlogs Impact](figures/backlogs_impact.png)

---

## 5. Compensation Tier Segmentation
Placed students are segmented across three distinct corporate compensation tiers:

![Salary Distribution by Tier](figures/salary_distribution_by_tier.png)

- **Tier 1 (Product / MNC):** Median ₹12.5 LPA (Top bracket up to ₹24 LPA)
- **Tier 2 (Growth / Mid-sized Tech):** Median ₹7.2 LPA
- **Tier 3 (Service / Mass Recruiters):** Median ₹4.1 LPA

---

## Strategic Recommendations for Career Development Cells
1. **Targeted Coding Intervention:** Introduce structured DSA training from Year 3 to boost technical assessment percentiles.
2. **Backlog Clearance Sprints:** Institutional academic assistance to clear backlogs before final year placement season starts.
3. **Mandatory Internship Credits:** Internships serve as a high-value signal for Tier 1 and Tier 2 job offers.
