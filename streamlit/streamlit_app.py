import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Job Salaries - Data Explorer", layout="wide")


@st.cache_data
def load_data(file_source):
    return pd.read_csv(file_source)


def get_default_csv_path():
    return Path(__file__).resolve().parents[1] / "job_salary_prediction_dataset.csv"


def render_data_info(dataframe):
    buffer = io.StringIO()
    dataframe.info(buf=buffer)
    st.code(buffer.getvalue(), language="text")


def render_null_summary(dataframe):
    null_counts = dataframe.isnull().sum()
    null_percent = (null_counts / len(dataframe) * 100).round(2)
    summary = (
        pd.DataFrame(
            {
                "feature": dataframe.columns,
                "null_count": null_counts.values,
                "null_percent": null_percent.values,
            }
        )
        .sort_values(by="null_count", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(summary, use_container_width=True)
    st.bar_chart(summary.set_index("feature")["null_count"])


def detect_outliers_iqr(dataframe):
    numeric_df = dataframe.select_dtypes(include=np.number)
    if numeric_df.empty:
        return pd.DataFrame()
    q1 = numeric_df.quantile(0.25)
    q3 = numeric_df.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (numeric_df.lt(lower)) | (numeric_df.gt(upper))
    outlier_counts = outlier_mask.sum().astype(int)
    outlier_percent = (outlier_counts / len(dataframe) * 100).round(2)
    return (
        pd.DataFrame(
            {
                "feature": outlier_counts.index,
                "outlier_count": outlier_counts.values,
                "outlier_percent": outlier_percent.values,
                "lower_bound": lower.values.round(3),
                "upper_bound": upper.values.round(3),
            }
        )
        .sort_values(by="outlier_count", ascending=False)
        .reset_index(drop=True)
    )


st.title("Job Salaries Data Exploration")

uploaded_file = st.file_uploader("Upload CSV file (optional)", type=["csv"])
default_path = get_default_csv_path()

if uploaded_file is not None:
    data = load_data(uploaded_file)
elif default_path.exists():
    data = load_data(default_path)
else:
    st.error("CSV file not found. Upload a file to continue.")
    st.stop()

st.subheader("Dataset Preview")
st.write(f"Rows: {len(data):,} | Columns: {data.shape[1]}")
st.dataframe(data.head(20), use_container_width=True)

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("data.info()")
    render_data_info(data)

with right_col:
    st.subheader("data.describe(include='all')")
    st.dataframe(data.describe(include="all").transpose(), use_container_width=True)

st.subheader("Value Counts for Each Feature")
columns = list(data.columns)
selected_column = st.selectbox("Choose feature", columns)
value_counts_df = (
    data[selected_column]
    .value_counts(dropna=False)
    .rename_axis(selected_column)
    .reset_index(name="count")
)
st.dataframe(value_counts_df, use_container_width=True)

with st.expander("Show value counts for all features"):
    for feature in columns:
        st.markdown(f"**{feature}**")
        all_counts = (
            data[feature]
            .value_counts(dropna=False)
            .rename_axis(feature)
            .reset_index(name="count")
        )
        st.dataframe(all_counts, use_container_width=True)

st.subheader("Null Value Detection")
render_null_summary(data)

st.subheader("Outlier Detection (IQR Method)")
outlier_summary = detect_outliers_iqr(data)
if outlier_summary.empty:
    st.info("No numeric features available for outlier detection.")
else:
    st.dataframe(outlier_summary, use_container_width=True)
    st.bar_chart(outlier_summary.set_index("feature")["outlier_count"])

st.subheader("Skewness (Numeric Features)")
numeric_data = data.select_dtypes(include=np.number)
if numeric_data.empty:
    st.info("No numeric features available for skewness analysis.")
else:
    skewness_df = (
    numeric_data.skew()
    .rename("skewness")
    .reset_index()
    .rename(columns={"index": "feature"})
    .sort_values(by="skewness", ascending=False)
)
    skewness_df["distribution"] = np.select(
        [skewness_df["skewness"] > 1, skewness_df["skewness"] < -1],
        ["Right-skewed", "Left-skewed"],
        default="Approximately symmetric",
    )
    st.dataframe(skewness_df, use_container_width=True)
    st.bar_chart(skewness_df.set_index("feature")["skewness"])
