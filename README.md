# 💼 Tech Job Salary Analysis: Prediction & Clustering

## 📌 Project Overview

This project explored salary patterns across 12 tech job titles using a structured
dataset of industry professionals. The goal was twofold: build a machine learning
model to predict salary based on candidate attributes, and apply clustering techniques
to discover natural groupings within the tech labor market.

---

## 📂 Dataset

The dataset used was `job_salary_prediction_dataset`, which contained the following features:

| Feature | Description |
|---|---|
| `job_title` | One of 12 tech roles (e.g., Data Scientist, ML Engineer) |
| `experience_years` | Total years of professional experience |
| `education_level` | Highest academic degree attained |
| `skills_count` | Number of technical skills listed |
| `industry` | Industry sector of the employer |
| `company_size` | Size category of the employing company |
| `location` | Geographic location of the role |
| `remote_work` | Whether the role was remote, hybrid, or on-site |
| `certifications` | Number or presence of professional certifications |
| `salary` | Target variable — annual compensation in USD |

---

## 🔍 Exploratory Data Analysis (EDA)

Before modeling, a thorough EDA phase was conducted to understand the data's structure
and distribution:

- **Salary distributions** were examined per job title to identify role-based compensation gaps.
- **Correlation analysis** was performed between numerical features (`experience_years`,
  `skills_count`, `certifications`) and the target variable `salary`.
- **Education level breakdowns** were visualized to assess whether advanced degrees
  consistently translated to higher pay across roles.
- **Remote work patterns** were analyzed in relation to location and salary, revealing
  geographic salary premiums that remote work partially neutralized.
- **Industry and company size** were explored as categorical salary drivers,
  with certain industries (e.g., Finance, Healthcare) consistently offering higher
  compensation.

---

## ⚙️ Feature Engineering

Several preprocessing and feature engineering steps were applied:

- Categorical variables (`job_title`, `education_level`, `industry`, `company_size`,
  `location`, `remote_work`) were encoded using label encoding and one-hot encoding
  where appropriate.
- Numerical features were scaled using **StandardScaler** to normalize the range of
  values before feeding them into distance-sensitive models.
- Multicollinearity between features was checked and low-variance features were
  considered for removal to reduce noise.

---

## 🤖 Salary Prediction Model

### Approach
A supervised regression pipeline was built to predict the `salary` target variable.
Multiple algorithms were trained and evaluated:

- **Linear Regression** — served as the baseline model.
- **Random Forest Regressor** — handled non-linear relationships and feature interactions well.
- **Gradient Boosting (XGBoost / LightGBM)** — delivered the strongest predictive performance.

### Evaluation Metrics
Models were evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Coefficient of Determination)

### Results
The best-performing model achieved a strong R² score, with `experience_years`,
`job_title`, and `industry` emerging as the top three salary predictors according
to feature importance analysis.

---

## 🔵 Clustering Analysis

### Approach
An unsupervised clustering analysis was performed to discover natural groupings
among professionals in the dataset without relying on job title labels.

- **K-Means Clustering** was the primary algorithm used.
- The optimal number of clusters was determined using the **Elbow Method** and
  validated with the **Silhouette Score**.
- Clusters were formed on features such as `experience_years`, `skills_count`,
  `education_level`, `certifications`, and `salary`.

### Findings
The clustering revealed distinct professional profiles, broadly corresponding to:

1. **Early-career generalists** — lower experience, moderate skills, entry-level salaries.
2. **Mid-level specialists** — solid experience, high skills count, above-average salaries.
3. **Senior technical experts** — high experience, certifications, top-tier compensation.
4. **Business-facing roles** — moderate technical depth, driven more by industry and company size.

These cluster profiles were cross-referenced against actual job titles to validate
that the unsupervised model captured meaningful real-world distinctions.

---

## 📊 Power BI Dashboard

A multi-page interactive Power BI dashboard was built to visualize the findings:

- **Overview Page** — Cross-role salary comparisons, top hiring industries, education
  distribution, remote work breakdown, and company size vs. salary.
- **Data & AI Page** — Deep-dive into AI Engineer, Data Analyst, Data Scientist,
  and Machine Learning Engineer roles.
- **Engineering Page** — Analysis of Cloud Engineer, DevOps, Software Engineer,
  Frontend, and Backend Developer salaries.
- **Business & Security Page** — Insights for Business Analyst, Product Manager,
  and Cybersecurity Analyst roles.

Custom DAX measures were written to power role-specific KPIs, salary gap calculations,
and filtered visuals scoped to each page's job title group.

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|---|---|
| Python (Pandas, Scikit-learn, XGBoost) | Data preprocessing, modeling, clustering |
| Matplotlib / Seaborn | Exploratory visualizations |
| Power BI Desktop | Interactive dashboard |
| DAX | Custom measures and calculated tables |
| Jupyter Notebook | Analysis and experimentation environment |

---

## 📁 Project Structure
├── data/
│   └── job_salary_prediction_dataset.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_salary_prediction.ipynb
│   └── 04_clustering.ipynb
├── models/
│   └── best_model.pkl
├── dashboard/
│   └── salary_dashboard.pbix
└── README.md

---

## ✅ Key Takeaways

- Experience and job title were the strongest individual predictors of salary.
- Remote work showed a geography-dependent effect — it narrowed salary gaps in
  high-cost locations but had less impact in mid-tier markets.
- Clustering successfully identified four distinct professional archetypes that
  cut across formal job title boundaries.
- The Power BI dashboard made the findings accessible and interactive for
  non-technical stakeholders.

---

## 👤 Author

**Youssef**  
Junior Data Analyst and Scientist 