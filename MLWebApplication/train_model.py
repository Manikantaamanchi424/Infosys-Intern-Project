"""
train_model.py
Train a RandomForestClassifier on the Iris dataset and save it as iris_model.joblib
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import pandas as pd


def main():
    # Load dataset
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    feature_names = list(X.columns)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: scaling + classifier
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}\n")
    print("Classification report:\n")
    print(classification_report(y_test, preds, target_names=data.target_names))

    # Save artifacts
    joblib.dump({
        "pipeline": pipeline,
        "feature_names": feature_names,
        "feature_stats": {
            "min": X.min().to_dict(),
            "max": X.max().to_dict(),
            "mean": X.mean().to_dict(),
            "std": X.std().to_dict()
        },
        "target_names": list(data.target_names)
    }, "iris_model.joblib")

    print("✅ Model saved to iris_model.joblib")


if __name__ == "__main__":
    main()
