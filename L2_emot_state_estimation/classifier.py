"""
Classifier selector / manager.

This module provides:
- `BaseClassifier` abstract interface for classifier implementations
- adapters for KNN / SVM / RandomForest
- `ClassifierManager` registry + selector to instantiate the chosen classifier

Prediction API:
- `predict(...)` -> label string
- `predict_with_confidence(...)` -> {"label": str, "confidence": float}
"""

import abc
import os
from pathlib import Path
from typing import Dict, List, Type
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from utils import plot_confusion_matrix


def _normalise_prediction_output(prediction) -> str:
    """Convert sklearn outputs to a stable label string."""
    if isinstance(prediction, np.ndarray):
        if prediction.size == 1:
            return str(prediction.item())
        return str(prediction.tolist())
    return str(prediction)


def _apply_class_balancing(
    X_train: list, y_train: list, method: str, random_state: int = 42
):
    """Apply class balancing to the training data."""
    if method == "none":
        return X_train, y_train

    elif method == "undersample":
        X_train = np.asarray(X_train, dtype=object)
        y_train = np.asarray(y_train)

        classes, counts = np.unique(y_train, return_counts=True)
        target_count = int(np.min(counts))
        rng = np.random.default_rng(random_state)

        sampled_indices = []
        for class_label in classes:
            class_indices = np.flatnonzero(y_train == class_label)
            chosen_indices = rng.choice(class_indices, size=target_count, replace=False)
            sampled_indices.extend(chosen_indices.tolist())

        sampled_indices = rng.permutation(sampled_indices)
        return X_train[sampled_indices].tolist(), y_train[sampled_indices].tolist()

    elif method == "oversample":
        X_train = np.asarray(X_train, dtype=object)
        y_train = np.asarray(y_train)

        classes, counts = np.unique(y_train, return_counts=True)
        target_count = int(np.max(counts))
        rng = np.random.default_rng(random_state)

        sampled_indices = []
        for class_label in classes:
            class_indices = np.flatnonzero(y_train == class_label)
            replace = len(class_indices) < target_count
            chosen_indices = rng.choice(
                class_indices, size=target_count, replace=replace
            )
            sampled_indices.extend(chosen_indices.tolist())

        sampled_indices = rng.permutation(sampled_indices)
        return X_train[sampled_indices].tolist(), y_train[sampled_indices].tolist()
    else:
        raise ValueError(f"Invalid class balancing method: {method}")


def _split_data(
    X: list, y: list, method: str, parameters: dict, people: List[int], stratify: list
) -> tuple[list, list, list, list]:
    """Split data into train/test sets based on the specified method."""
    if method == "random":
        test_size = parameters["random"]["test_size"]
        random_state = parameters["random"]["random_state"]
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    elif method == "subject":
        test_subjects = parameters["subject"]["test_subjects"]
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        people = np.asarray(people)

        test_mask = np.isin(people, test_subjects)
        X_train, X_test = X[~test_mask].tolist(), X[test_mask].tolist()
        y_train, y_test = y[~test_mask].tolist(), y[test_mask].tolist()

        return X_train, X_test, y_train, y_test

    else:
        raise ValueError(f"Invalid data split method: {method}")


class BaseClassifier(abc.ABC):
    """Abstract base for classifier implementations."""

    name: str = "base"

    def __init__(
        self,
        model_path: str = None,
        hyperparams: dict = None,
        num_classes: int = None,
        input_len: int = None,
    ):
        self.model = None
        self.model_path = model_path
        self.hyperparams = hyperparams
        self.num_classes = num_classes
        self.input_len = input_len

        if model_path:
            self.load_model()

    @abc.abstractmethod
    def fit(self, pow_vectors: List[List[float]], labels: List[str]):
        raise NotImplementedError()

    def predict(self, pow_vector: List[float]) -> str:
        if self.model is None:
            raise RuntimeError(f"{self.name} model is not loaded")

        features = self._prepare_features(pow_vector)
        prediction = self.model.predict(features)[0]
        return _normalise_prediction_output(prediction)

    def predict_with_confidence(self, pow_vector: List[float]) -> Dict:
        if self.model is None:
            raise RuntimeError(f"{self.name} model is not loaded")

        features = self._prepare_features(pow_vector)

        if hasattr(self.model, "predict_proba") and hasattr(self.model, "classes_"):
            probabilities = self.model.predict_proba(features)[0]
            class_index = int(np.argmax(probabilities))
            return {
                "label": _normalise_prediction_output(self.model.classes_[class_index]),
                "confidence": float(probabilities[class_index]),
            }

        prediction = self.model.predict(features)[0]
        return {
            "label": _normalise_prediction_output(prediction),
            "confidence": None,
        }

    def batch_predict(self, pow_vectors: List[List[float]]) -> List[str]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        features = np.asarray(pow_vectors, dtype=float)
        predictions = self.model.predict(features)
        return [_normalise_prediction_output(prediction) for prediction in predictions]

    def batch_predict_with_confidence(
        self, pow_vectors: List[List[float]]
    ) -> List[Dict]:
        return [self.predict_with_confidence(v) for v in pow_vectors]

    def load_model(self):
        loaded = joblib.load(self.model_path)
        model_metadata = loaded if isinstance(loaded, dict) else {}
        self.model = (
            loaded.get("model")
            if isinstance(loaded, dict) and "model" in loaded
            else loaded
        )
        # print(f"Loaded model from {self.model_path} with metadata: {model_metadata}")
        self.validate_model_metadata(model_metadata)
        return self.model

    def validate_existing_model(self, model_path: str):

        if os.path.exists(model_path):
            print(f"Existing model found at {model_path}; validating it.")
            loaded = joblib.load(model_path)
            model_metadata = loaded if isinstance(loaded, dict) else {}
            self.validate_model_metadata(model_metadata)

    def validate_model_metadata(self, metadata: dict):
        saved_input_len = metadata.get("input_len")
        saved_num_classes = metadata.get("num_classes")
        saved_classifier = metadata.get("classifier")
        saved_hyperparams = metadata.get("hyperparams")

        if saved_input_len is not None and int(saved_input_len) != self.input_len:
            print(
                f"Model input size mismatch: expected {self.input_len}, got {saved_input_len}"
            )

        if saved_num_classes is not None and int(saved_num_classes) != self.num_classes:
            print(
                f"Model output size mismatch: expected {self.num_classes}, got {saved_num_classes}"
            )

        if saved_classifier is not None and saved_classifier != self.name:
            print(
                f"Model classifier mismatch: expected {self.name}, got {saved_classifier}"
            )

        for key, expected_value in self.hyperparams.items():
            actual_value = saved_hyperparams.get(key)
            if actual_value != expected_value:
                print(
                    f"Loaded model hyperparam mismatch for '{key}': "
                    f"expected {expected_value}, got {actual_value}"
                )

    def save_model(self, model_path: str):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        joblib.dump(
            {
                "model": self.model,
                "classifier": self.name,
                "hyperparams": self.hyperparams,
                "num_classes": self.num_classes,
                "input_len": self.input_len,
            },
            model_path,
        )
        return model_path

    def _prepare_features(self, pow_vector: List[float]) -> np.ndarray:
        return np.asarray(pow_vector, dtype=float).reshape(1, -1)


class KNNClassifierAdapter(BaseClassifier):
    name = "knn"

    def __init__(
        self,
        model_path: str = None,
        hyperparams: dict = None,
        num_classes: int = None,
        input_len: int = None,
    ):
        super().__init__(
            model_path=model_path,
            hyperparams=hyperparams["knn"],
            num_classes=num_classes,
            input_len=input_len,
        )

    def fit(self, pow_vectors: List[List[float]], labels: List[str]):
        self.model = KNeighborsClassifier(**self.hyperparams)
        self.model.fit(np.asarray(pow_vectors, dtype=float), np.asarray(labels))
        return self.model


class SVMClassifierAdapter(BaseClassifier):
    name = "svm"

    def __init__(
        self,
        model_path: str = None,
        hyperparams: dict = None,
        num_classes: int = None,
        input_len: int = None,
    ):
        super().__init__(
            model_path=model_path,
            hyperparams=hyperparams["svm"],
            num_classes=num_classes,
            input_len=input_len,
        )

    def fit(self, pow_vectors: List[List[float]], labels: List[str]):
        fit_params = dict(self.hyperparams)
        fit_params.setdefault("probability", True)

        self.model = SVC(**fit_params)
        self.model.fit(np.asarray(pow_vectors, dtype=float), np.asarray(labels))
        return self.model


class RFClassifierAdapter(BaseClassifier):
    name = "random_forest"

    def __init__(
        self,
        model_path: str = None,
        hyperparams: dict = None,
        num_classes: int = None,
        input_len: int = None,
    ):
        super().__init__(
            model_path=model_path,
            hyperparams=hyperparams["random_forest"],
            num_classes=num_classes,
            input_len=input_len,
        )

    def fit(self, pow_vectors: List[List[float]], labels: List[str]):
        self.model = RandomForestClassifier(**self.hyperparams)
        self.model.fit(np.asarray(pow_vectors, dtype=float), np.asarray(labels))
        return self.model


class ClassifierManager:
    """Registry + selector for classifier implementations."""

    _registry: Dict[str, Type[BaseClassifier]] = {}

    def __init__(self, input_len: int):
        self.input_len = input_len
        self.active: BaseClassifier = None

        self.register(KNNClassifierAdapter.name, KNNClassifierAdapter)
        self.register(SVMClassifierAdapter.name, SVMClassifierAdapter)
        self.register(RFClassifierAdapter.name, RFClassifierAdapter)

    @classmethod
    def register(cls, name: str, impl: Type[BaseClassifier]):
        cls._registry[name] = impl

    def load(
        self,
        model_path: str,
        hyperparams: dict,
        num_classes: int,
        **kwargs,
    ):
        loaded = joblib.load(model_path)
        if not isinstance(loaded, dict) or "classifier" not in loaded:
            raise ValueError("Saved model metadata missing classifier name")

        name = loaded["classifier"]
        impl = self._registry.get(name)
        if impl is None:
            raise ValueError(f"Classifier '{name}' not registered")

        self.active = impl(
            model_path=model_path,
            hyperparams=hyperparams,
            num_classes=num_classes,
            input_len=self.input_len,
            **kwargs,
        )
        return self.active

    def train(
        self,
        name: str,
        pow_vectors: List[List[float]],
        labels: List[str],
        model_path: str,
        hyperparams: dict,
        num_classes: int,
        class_balancing: str,
        data_split_method: str,
        data_split_parameters: dict,
        people: List[int],
        stratify: bool = True,
        **kwargs,
    ):
        impl = self._registry.get(name)
        if impl is None:
            raise ValueError(f"Classifier '{name}' not registered")

        self.active = impl(
            model_path=None,
            hyperparams=hyperparams,
            num_classes=num_classes,
            input_len=self.input_len,
            **kwargs,
        )

        stratify_labels = labels if stratify else None

        X_train, X_test, y_train, y_test = _split_data(
            pow_vectors,
            labels,
            data_split_method,
            data_split_parameters,
            people,
            stratify_labels,
        )

        print(f"X_train size: {len(X_train)}, X_test size: {len(X_test)}")

        X_train, y_train = _apply_class_balancing(
            X_train, y_train, method=class_balancing
        )

        print(f"X_train size: {len(X_train)}, X_test size: {len(X_test)}")

        self.active.fit(X_train, y_train)
        y_pred = self.active.batch_predict(X_test)

        if model_path:
            self.active.save_model(model_path)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        plot_confusion_matrix(y_test, y_pred)
        return self.active

    def predict(self, pow_vector: List[float]) -> str:
        if not self.active:
            raise RuntimeError("No classifier selected")
        return self.active.predict(pow_vector)

    def predict_with_confidence(self, pow_vector: List[float]) -> Dict:
        if not self.active:
            raise RuntimeError("No classifier selected")
        return self.active.predict_with_confidence(pow_vector)

    def batch_predict(self, pow_vectors: List[List[float]]) -> List[str]:
        if not self.active:
            raise RuntimeError("No classifier selected")
        return self.active.batch_predict(pow_vectors)

    def batch_predict_with_confidence(
        self, pow_vectors: List[List[float]]
    ) -> List[Dict]:
        if not self.active:
            raise RuntimeError("No classifier selected")
        return self.active.batch_predict_with_confidence(pow_vectors)


__all__ = ["BaseClassifier", "ClassifierManager"]
