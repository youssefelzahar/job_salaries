import streamlit as st

from app_pages import cluster, explore, predict
from app_pages.utils import get_default_csv_path, load_data

st.set_page_config(page_title="Job Salaries", layout="wide")

st.title("Job Salaries")

mode = st.sidebar.radio("Mode", ["Explore Data", "Predict Salary", "Cluster (Numeric)"], index=0)

uploaded_file = st.file_uploader("Upload CSV file (optional)", type=["csv"])
default_path = get_default_csv_path()

if uploaded_file is not None:
    data = load_data(uploaded_file)
elif default_path.exists():
    data = load_data(default_path)
else:
    st.error("CSV file not found. Upload a file to continue.")
    st.stop()

st.sidebar.write(f"Rows: {len(data):,}")
st.sidebar.write(f"Columns: {data.shape[1]}")

if mode == "Explore Data":
    explore.render(data)
elif mode == "Predict Salary":
    predict.render(data)
else:
    cluster.render(data)
