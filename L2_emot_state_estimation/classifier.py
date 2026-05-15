"""
Classifier selector / manager.

This module provides:
- adapters for KNN / SVM / RandomForest through a simple factory
- ClassifierManager to train, save, load, and run predictions

Prediction API:
- predict(...) -> label string
- predict_with_confidence(...) -> {"label": str, "confidence": float | None}
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from utils import plot_confusion_matrix


CLASSIFIERS = {
    "knn": KNeighborsClassifier,
    "svm": SVC,
    "random_forest": RandomForestClassifier,
}


def _build_estimator(name: str, hyperparams: dict[str, Any]):
    if name not in CLASSIFIERS:
        raise ValueError(f"Classifier '{name}' not registered")

    params = dict(hyperparams or {})

    # SVM needs probability=True to enable confidence outputs
    if name == "svm":
        params.setdefault("probability", True)

    return CLASSIFIERS[name](**params)


def _split_data(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    parameters: dict,
    people: list[int] | None,  # Optional
    stratify: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/test sets based on the specified method."""
    if method == "random":
        return train_test_split(
            X,
            y,
            test_size=parameters["random"]["test_size"],
            random_state=parameters["random"]["random_state"],
            stratify=y if stratify else None,
        )

    if method == "subject":
        people = np.asarray(people)
        test_subjects = parameters["subject"]["test_subjects"]
        test_mask = np.isin(people, test_subjects)
        return X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]

    raise ValueError(f"Invalid data split method: {method}")


def _apply_class_balancing(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply simple over/under-sampling to the training data."""
    if method == "none":
        return X, y

    classes, counts = np.unique(y, return_counts=True)
    rng = np.random.default_rng(random_state)

    if method == "undersample":
        target_count = int(np.min(counts))
    elif method == "oversample":
        target_count = int(np.max(counts))
    else:
        raise ValueError(f"Invalid class balancing method: {method}")

    sampled_indices = []

    for class_label in classes:
        class_indices = np.flatnonzero(y == class_label)
        replace = method == "oversample" and len(class_indices) < target_count
        chosen_indices = rng.choice(
            class_indices,
            size=target_count,
            replace=replace,
        )
        sampled_indices.append(chosen_indices)

    sampled_indices = np.concatenate(sampled_indices)
    sampled_indices = rng.permutation(sampled_indices)

    return X[sampled_indices], y[sampled_indices]


class ClassifierManager:
    """Train, save, load, and use one of the supported sklearn classifiers."""

    def __init__(self, input_len: int):
        self.input_len = input_len
        self.active_model = None
        self.active_name: str | None = None
        self.active_hyperparams: dict[str, Any] | None = None
        self.active_num_classes: int | None = None

    def _ensure_active(self):
        if self.active_model is None:
            raise RuntimeError("No classifier selected")

    def _warn_on_metadata_mismatch(
        self,
        metadata: dict[str, Any],
        expected_num_classes: int | None,
    ):
        saved_input_len = metadata.get("input_len")
        saved_num_classes = metadata.get("num_classes")

        if (
            saved_input_len is not None
            and self.input_len is not None
            and int(saved_input_len) != int(self.input_len)
        ):
            warnings.warn(
                "Loaded model input length does not match current manager input "
                f"length: saved={saved_input_len}, expected={self.input_len}",
                stacklevel=2,
            )

        if (
            saved_num_classes is not None
            and expected_num_classes is not None
            and int(saved_num_classes) != int(expected_num_classes)
        ):
            warnings.warn(
                "Loaded model output class count does not match expected class "
                f"count: saved={saved_num_classes}, expected={expected_num_classes}",
                stacklevel=2,
            )

    def save_model(self, model_path: str):
        self._ensure_active()

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.active_model,
            "classifier": self.active_name,
            "hyperparams": self.active_hyperparams,
            "input_len": self.input_len,
            "num_classes": self.active_num_classes,
        }

        joblib.dump(payload, model_path)
        return model_path

    def load(self, model_path: str, num_classes: int | None = None):
        payload = joblib.load(model_path)

        model = payload["model"]
        classifier_name = payload["classifier"]
        hyperparams = payload.get("hyperparams", {})

        if classifier_name not in CLASSIFIERS:
            raise ValueError(f"Classifier '{classifier_name}' not registered")

        self._warn_on_metadata_mismatch(payload, num_classes)

        self.active_model = model
        self.active_name = classifier_name
        self.active_hyperparams = hyperparams
        self.active_num_classes = payload.get("num_classes", num_classes)

        return self.active_model

    def train(
        self,
        name: str,
        pow_vectors: list[list[float]],
        labels: list[str],
        model_path: str,
        hyperparams: dict[str, Any],
        num_classes: int,
        class_balancing: str,
        data_split_method: str,
        data_split_parameters: dict,
        people: list[int],
        stratify: bool = True,
    ):
        X = np.asarray(pow_vectors, dtype=float)
        y = np.asarray(labels)

        X_train, X_test, y_train, y_test = _split_data(
            X,
            y,
            data_split_method,
            data_split_parameters,
            people,
            stratify,
        )

        print(f"X_train size: {len(X_train)}, X_test size: {len(X_test)}")

        X_train, y_train = _apply_class_balancing(
            X_train,
            y_train,
            method=class_balancing,
        )

        print(
            f"X_train size after balancing: {len(X_train)}, X_test size: {len(X_test)}"
        )

        model = _build_estimator(name, hyperparams)
        model.fit(X_train, y_train)

        self.active_model = model
        self.active_name = name
        self.active_hyperparams = hyperparams
        self.active_num_classes = num_classes

        y_pred = self.batch_predict(X_test)

        if model_path:
            self.save_model(model_path)

        return {
            "model": self.active_model,
            "y_test": y_test,
            "y_pred": np.asarray(y_pred),
        }

    def predict(self, pow_vector: list[float]) -> str:
        return self.batch_predict([pow_vector])[0]

    def predict_with_confidence(self, pow_vector: list[float]) -> dict[str, Any]:
        return self.batch_predict_with_confidence([pow_vector])[0]

    def batch_predict(self, pow_vectors: list[list[float]] | np.ndarray) -> list[str]:
        self._ensure_active()

        model = self.active_model
        features = np.asarray(pow_vectors, dtype=float)
        return [str(prediction) for prediction in model.predict(features)]

    def batch_predict_with_confidence(
        self, pow_vectors: list[list[float]]
    ) -> list[dict[str, Any]]:
        self._ensure_active()

        model = self.active_model
        features = np.asarray(pow_vectors, dtype=float)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            class_indices = np.argmax(probabilities, axis=1)

            labels = model.classes_[class_indices]
            confidences = probabilities[np.arange(len(probabilities)), class_indices]
            return [
                {"label": str(label), "confidence": float(confidence)}
                for label, confidence in zip(labels, confidences)
            ]

        predictions = model.predict(features)
        return [
            {"label": str(prediction), "confidence": None} for prediction in predictions
        ]


__all__ = ["ClassifierManager"]
