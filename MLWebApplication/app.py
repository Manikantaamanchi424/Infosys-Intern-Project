"""
app.py
Streamlit app for Iris dataset classification and exploration
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris Classifier", layout="wide")


@st.cache_data
def load_model_artifacts(path="iris_model.joblib"):
    return joblib.load(path)


def prediction_ui(artifacts):
    st.title("🌸 Iris Flower Classifier")
    st.write("Move the sliders to input flower features and predict the species.")

    feature_names = artifacts["feature_names"]
    stats = artifacts["feature_stats"]
    model = artifacts["pipeline"]
    target_names = artifacts["target_names"]

    # Arrange inputs in columns
    col1, col2 = st.columns(2)
    inputs = {}

    with col1:
        for fname in feature_names[:2]:
            min_v = float(stats["min"][fname])
            max_v = float(stats["max"][fname])
            mean_v = float(stats["mean"][fname])
            inputs[fname] = st.slider(
                label=fname.replace("_", " ").title(),
                min_value=round(min_v - 1.0, 2),
                max_value=round(max_v + 1.0, 2),
                value=round(mean_v, 2),
                step=0.01,
                help=f"Range: {min_v} — {max_v}"
            )

    with col2:
        for fname in feature_names[2:]:
            min_v = float(stats["min"][fname])
            max_v = float(stats["max"][fname])
            mean_v = float(stats["mean"][fname])
            inputs[fname] = st.slider(
                label=fname.replace("_", " ").title(),
                min_value=round(min_v - 1.0, 2),
                max_value=round(max_v + 1.0, 2),
                value=round(mean_v, 2),
                step=0.01,
                help=f"Range: {min_v} — {max_v}"
            )

    # Predict button
    if st.button("🔮 Predict"):
        X_input = np.array([list(inputs.values())])
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        pred_name = target_names[pred]

        # Show result with confidence
        if proba[pred] > 0.8:
            st.success(f"Prediction: **{pred_name}** (confidence: {proba[pred]:.2f})")
        elif proba[pred] > 0.5:
            st.info(f"Prediction: **{pred_name}** (confidence: {proba[pred]:.2f})")
        else:
            st.warning(f"Prediction: **{pred_name}** (confidence: {proba[pred]:.2f})")

        # Show probability table
        prob_df = pd.DataFrame({"Species": target_names, "Probability": proba})
        st.table(prob_df.style.format({"Probability": "{:.2f}"}))


def exploration_ui(df):
    st.title("📊 Dataset Exploration")
    st.write("Explore simple visualizations of the Iris dataset.")

    plot_type = st.sidebar.selectbox("Choose Plot Type", ["Histogram", "Scatter"])

    if plot_type == "Histogram":
        feature = st.selectbox("Feature", df.columns[:-1])
        bins = st.slider("Bins", 5, 50, 15)

        fig, ax = plt.subplots()
        ax.hist(df[feature], bins=bins)
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        ax.set_title(f"Histogram of {feature}")
        st.pyplot(fig)

    else:  # Scatter
        x_feature = st.selectbox("X-axis", df.columns[:-1], index=0)
        y_feature = st.selectbox("Y-axis", df.columns[:-1], index=1)

        fig, ax = plt.subplots()
        for species in df["species"].unique():
            subset = df[df["species"] == species]
            ax.scatter(subset[x_feature], subset[y_feature], label=species)
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.legend()
        ax.set_title(f"{y_feature} vs {x_feature}")
        st.pyplot(fig)


def main():
    try:
        artifacts = load_model_artifacts("iris_model.joblib")
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}. Run train_model.py first.")
        return

    # Sidebar toggle
    mode = st.sidebar.radio("Mode", ("Prediction", "Exploration"))

    # Load dataset
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns={c: c.replace(" ", "_") for c in df.columns})
    df["species"] = [data.target_names[i] for i in df["target"]]
    df = df.drop(columns=["target"])

    if mode == "Prediction":
        prediction_ui(artifacts)
    else:
        exploration_ui(df)


if __name__ == "__main__":
    main()
