import streamlit as st

from app_pages.utils import compute_percentile_ranking, load_salary_bundle, predict_salary_with_ci


def render(data):
    st.subheader("Salary Prediction")

    salary_bundle = load_salary_bundle()
    if salary_bundle is None:
        st.error("Saved salary model not found. Run the notebook to create save_models/salary_model.pkl.")
        st.stop()

    feature_cols = salary_bundle.get("feature_cols") or ["certifications", "skills_count", "experience_years"]
    for col in feature_cols + ["salary"]:
        if col not in data.columns:
            st.error(f"Missing column in dataset: {col}")
            st.stop()

    feature_stats = data[feature_cols].agg(["min", "max"]).transpose()
    cert_min, cert_max = int(feature_stats.loc["certifications", "min"]), int(feature_stats.loc["certifications", "max"])
    skills_min, skills_max = int(feature_stats.loc["skills_count", "min"]), int(feature_stats.loc["skills_count", "max"])
    exp_min, exp_max = int(feature_stats.loc["experience_years", "min"]), int(feature_stats.loc["experience_years", "max"])

    left_col, right_col = st.columns(2)
    with left_col:
        certifications = st.slider("certifications", min_value=cert_min, max_value=cert_max, value=min(2, cert_max))
        skills_count = st.slider("skills_count", min_value=skills_min, max_value=skills_max, value=min(10, skills_max))
        experience_years = st.slider("experience_years", min_value=exp_min, max_value=exp_max, value=min(5, exp_max))
        confidence_pct = st.slider("Confidence level (%)", min_value=80, max_value=99, value=95, step=1)

    with right_col:
        st.write("Model features:")
        st.write(feature_cols)

    confidence_level = confidence_pct / 100.0
    prediction, ci, residual_std = predict_salary_with_ci(
        data=data,
        salary_bundle=salary_bundle,
        certifications=certifications,
        skills_count=skills_count,
        experience_years=experience_years,
        confidence_level=confidence_level,
    )

    percentile = compute_percentile_ranking(data["salary"], prediction)

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Prediction", f"${prediction:,.0f}")
    with kpi2:
        st.metric("Confidence Interval", f"${ci[0]:,.0f} to ${ci[1]:,.0f}")
    with kpi3:
        st.metric("Percentile Ranking", f"{percentile:.2f}%")

    st.subheader("Model Error Reference")
    st.write(f"Residual STD used for CI: ${residual_std:,.0f}")

    st.subheader("Salary Distribution Reference")
    q10, q50, q90 = data["salary"].quantile([0.1, 0.5, 0.9]).tolist()
    st.write(f"10th: ${q10:,.0f} | Median: ${q50:,.0f} | 90th: ${q90:,.0f}")
