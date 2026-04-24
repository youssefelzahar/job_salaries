import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def load_data(file_source):
    return pd.read_csv(file_source)


def get_default_csv_path():
    return Path(__file__).resolve().parents[1] / "job_salary_prediction_dataset.csv"


def _load_pickle_or_joblib(model_path: Path):
    try:
        import joblib

        return joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            return pickle.load(f)


@st.cache_resource
def load_salary_bundle():
    model_path = Path(__file__).resolve().parents[1] / "save_models" / "salary_model.pkl"
    if not model_path.exists():
        return None
    obj = _load_pickle_or_joblib(model_path)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj, "feature_cols": ["certifications", "skills_count", "experience_years"], "residual_std": None}


@st.cache_resource
def load_kmeans_bundle():
    model_path = Path(__file__).resolve().parents[1] / "save_models" / "kmeans_model.pkl"
    if not model_path.exists():
        return None
    obj = _load_pickle_or_joblib(model_path)
    if isinstance(obj, dict):
        if "pipeline" in obj:
            return obj
        if "model" in obj:
            return {
                "pipeline": obj["model"],
                "feature_cols": obj.get("feature_cols") or ["experience_years", "skills_count", "certifications"],
                "n_clusters": obj.get("n_clusters"),
            }
    return {"pipeline": obj, "feature_cols": ["experience_years", "skills_count", "certifications"], "n_clusters": None}


def compute_percentile_ranking(series, value):
    series = pd.Series(series, dtype=float)
    return float((series <= float(value)).mean() * 100.0)


def _z_value(confidence_level):
    z_table = {
        0.80: 1.281551565545,
        0.85: 1.439531470939,
        0.90: 1.644853626951,
        0.95: 1.959963984540,
        0.98: 2.326347874041,
        0.99: 2.575829303549,
    }
    confidence_level = float(confidence_level)
    if confidence_level in z_table:
        return z_table[confidence_level]
    keys = sorted(z_table.keys())
    if confidence_level <= keys[0]:
        return z_table[keys[0]]
    if confidence_level >= keys[-1]:
        return z_table[keys[-1]]
    for low, high in zip(keys, keys[1:]):
        if low <= confidence_level <= high:
            w = (confidence_level - low) / (high - low)
            return (1.0 - w) * z_table[low] + w * z_table[high]
    return 1.959963984540


def predict_salary_with_ci(data, salary_bundle, certifications, skills_count, experience_years, confidence_level):
    model = salary_bundle["model"]
    feature_cols = salary_bundle.get("feature_cols") or ["certifications", "skills_count", "experience_years"]
    x0 = pd.DataFrame(
        [[float(certifications), float(skills_count), float(experience_years)]],
        columns=feature_cols,
    )
    prediction = float(model.predict(x0)[0])

    residual_std = salary_bundle.get("residual_std")
    if residual_std is None:
        x_all = data[feature_cols]
        y_all = data["salary"].astype(float)
        y_hat = pd.Series(model.predict(x_all), index=y_all.index, dtype=float)
        residual_std = float((y_all - y_hat).std(ddof=0))

    z = _z_value(confidence_level)
    lower = prediction - z * residual_std
    upper = prediction + z * residual_std

    return prediction, (lower, upper), residual_std
