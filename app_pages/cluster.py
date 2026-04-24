import numpy as np
import pandas as pd
import streamlit as st

from app_pages.utils import load_kmeans_bundle


def _try_import_sklearn():
    try:
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score

        return PCA, silhouette_score
    except Exception:
        return None, None


def render(data):
    st.subheader("Clustering (Saved KMeans Model)")

    kmeans_bundle = load_kmeans_bundle()
    if kmeans_bundle is None:
        st.error("Saved KMeans model not found. Run the notebook to create save_models/kmeans_model.pkl.")
        st.stop()

    pipeline = kmeans_bundle.get("pipeline")
    feature_cols = kmeans_bundle.get("feature_cols") or []
    if pipeline is None or not hasattr(pipeline, "predict"):
        st.error("Saved KMeans bundle is missing a usable clustering pipeline.")
        st.stop()

    estimator_name = None
    if hasattr(pipeline, "named_steps"):
        estimator = pipeline.named_steps.get("model")
        if estimator is not None:
            estimator_name = estimator.__class__.__name__.lower()
    if estimator_name is not None and "kmeans" not in estimator_name:
        st.error("save_models/kmeans_model.pkl does not contain a KMeans pipeline. Recreate the clustering model file.")
        st.stop()

    if not feature_cols:
        st.error("Saved KMeans bundle is missing feature_cols.")
        st.stop()

    missing_cols = [c for c in feature_cols if c not in data.columns]
    if missing_cols:
        st.error(f"Dataset is missing required columns: {missing_cols}")
        st.stop()

    clean_df = data[feature_cols].dropna().copy()
    if len(clean_df) < 3:
        st.warning("Not enough rows after dropping nulls to run clustering.")
        st.stop()

    x = clean_df.to_numpy(dtype=float)
    labels = pipeline.predict(clean_df)
    labels = np.asarray(labels, dtype=int)

    result_df = clean_df.copy()
    result_df["cluster"] = labels

    st.subheader("Cluster Sizes")
    cluster_sizes = result_df["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(name="count")
    st.dataframe(cluster_sizes, use_container_width=True)
    st.bar_chart(cluster_sizes.set_index("cluster")["count"])

    st.subheader("Cluster Profile (Means)")
    profile = result_df.groupby("cluster")[feature_cols].mean().round(3)
    profile["count"] = result_df.groupby("cluster").size()
    st.dataframe(profile.reset_index(), use_container_width=True)

    PCA, silhouette_score = _try_import_sklearn()
    if silhouette_score is not None:
        x_for_score = x
        try:
            if hasattr(pipeline, "named_steps") and "scaler" in pipeline.named_steps:
                x_for_score = pipeline.named_steps["scaler"].transform(x_for_score)
        except Exception:
            x_for_score = x

        score = float("nan")
        if len(np.unique(labels)) > 1 and len(clean_df) > len(np.unique(labels)):
            try:
                score = float(silhouette_score(x_for_score, labels))
            except Exception:
                score = float("nan")
        st.metric("Silhouette Score", f"{score:.4f}" if np.isfinite(score) else "N/A")

    show_pca = st.checkbox("Show 2D PCA plot", value=True)
    if show_pca and PCA is not None and x.shape[1] >= 2:
        try:
            x_plot = x
            if hasattr(pipeline, "named_steps") and "scaler" in pipeline.named_steps:
                x_plot = pipeline.named_steps["scaler"].transform(x_plot)
            pca = PCA(n_components=2, random_state=42)
            xy = pca.fit_transform(x_plot)
            plot_df = pd.DataFrame(
                {"x": xy[:, 0], "y": xy[:, 1], "cluster": result_df["cluster"].astype(str).values}
            )
            st.vega_lite_chart(
                plot_df,
                {
                    "mark": {"type": "circle", "opacity": 0.7},
                    "encoding": {
                        "x": {"field": "x", "type": "quantitative"},
                        "y": {"field": "y", "type": "quantitative"},
                        "color": {"field": "cluster", "type": "nominal"},
                        "tooltip": [{"field": "cluster"}, {"field": "x"}, {"field": "y"}],
                    },
                },
                use_container_width=True,
            )
        except Exception:
            st.info("PCA plot unavailable for this saved model.")
