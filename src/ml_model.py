"""Experimental ML model for the Class Quality Assessor.

Requires human-labeled lesson examples.
Predictions support review; they are not definitive teaching grades.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "duration_minutes",
    "introduction_present",
    "question_marker_count",
    "homework_present",
    "closure_present",
    "average_brightness",
    "dark_frame_ratio",
    "black_frame_ratio",
    "face_detection_ratio",
]

TARGET = "human_label"


def train_model(csv_path, model_path="models/lesson_quality_model.joblib"):
    """Train and evaluate using a human-labeled CSV."""

    data = pd.read_csv(csv_path)

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    data = data[required].copy()
    data[TARGET] = data[TARGET].astype("string").str.strip()
    data = data.dropna(subset=[TARGET])
    data = data[data[TARGET] != ""]

    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Rows with no usable feature information cannot train the model.
    data = data.dropna(subset=FEATURES, how="all")

    if len(data) < 20:
        raise ValueError(
            f"Only {len(data)} usable labeled lessons. "
            "Collect at least 20 to run this initial experiment; "
            "more examples are strongly recommended."
        )

    class_counts = data[TARGET].value_counts()

    if len(class_counts) < 2:
        raise ValueError("At least two different human labels are required.")

    if class_counts.min() < 2:
        raise ValueError(
            "Each label needs at least two examples for a stratified split."
        )

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "training_examples": len(X_train),
        "test_examples": len(X_test),
        "class_counts": class_counts.to_dict(),
        "accuracy": round(accuracy_score(y_test, predictions), 3),
        "balanced_accuracy": round(
            balanced_accuracy_score(y_test, predictions), 3
        ),
        "classification_report": classification_report(
            y_test,
            predictions,
            zero_division=0,
        ),
    }

    # Save only after training and evaluation complete.
    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, output_path)

    return metrics


def predict_lesson(features, model_path="models/lesson_quality_model.joblib"):
    """Predict a human-label category for one lesson's feature dictionary."""

    saved = joblib.load(model_path)
    model = saved["model"]
    columns = saved["features"]

    row = pd.DataFrame(
        [{column: features.get(column) for column in columns}],
        columns=columns,
    )

    prediction = model.predict(row)[0]

    return {
        "predicted_label": str(prediction),
        "human_review_required": True,
        "note": (
            "Experimental prediction based on human-labeled examples. "
            "Review the lesson before making decisions."
        ),
    }
