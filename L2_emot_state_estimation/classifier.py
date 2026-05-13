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

    @abc.abstractmethod
    def predict(self, pow_vector: List[float]) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def predict_with_confidence(self, pow_vector: List[float]) -> Dict:
        raise NotImplementedError()

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
        print(
            f"Loaded model from {self.model_path} with metadata: {model_metadata}"
        )
        self.validate_model_metadata(model_metadata)
        return self.model

    def validate_existing_model(self, model_path: str) -> bool:

        if os.path.exists(model_path):
            print(f"Existing model found at {model_path}; validating it.")
            loaded = joblib.load(model_path)
            model_metadata = loaded if isinstance(loaded, dict) else {}
            self.validate_model_metadata(model_metadata)
            return True

        return False

    def validate_model_metadata(self, metadata: dict):
        saved_input_len = metadata.get("input_len")
        saved_num_classes = metadata.get("num_classes")

        if saved_input_len is not None and int(saved_input_len) != self.input_len:
            raise ValueError(
                f"Model input size mismatch: expected {self.input_len}, got {saved_input_len}"
            )

        if saved_num_classes is not None and int(saved_num_classes) != self.num_classes:
            raise ValueError(
                f"Model output size mismatch: expected {self.num_classes}, got {saved_num_classes}"
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
        if self.model is None:
            self.model = KNeighborsClassifier(**self.hyperparams)
        self.model.fit(np.asarray(pow_vectors, dtype=float), np.asarray(labels))
        return self.model

    def predict(self, pow_vector: List[float]) -> str:
        if self.model is None:
            raise RuntimeError("KNN model is not loaded")

        features = self._prepare_features(pow_vector)
        prediction = self.model.predict(features)[0]
        return _normalise_prediction_output(prediction)

    def predict_with_confidence(self, pow_vector: List[float]) -> Dict:
        if self.model is None:
            raise RuntimeError("KNN model is not loaded")

        features = self._prepare_features(pow_vector)
        label = self.predict(pow_vector)
        confidence = 0.65

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))

            if hasattr(self.model, "classes_"):
                class_index = int(np.argmax(probabilities))
                label = _normalise_prediction_output(self.model.classes_[class_index])

        return {
            "label": label,
            "confidence": confidence,
        }


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

        if self.model is None:
            self.model = SVC(**fit_params)
        self.model.fit(np.asarray(pow_vectors, dtype=float), np.asarray(labels))
        return self.model

    def predict(self, pow_vector: List[float]) -> str:
        if self.model is None:
            raise RuntimeError("SVM model is not loaded")

        features = self._prepare_features(pow_vector)
        prediction = self.model.predict(features)[0]
        return _normalise_prediction_output(prediction)

    def predict_with_confidence(self, pow_vector: List[float]) -> Dict:
        if self.model is None:
            raise RuntimeError("SVM model is not loaded")

        features = self._prepare_features(pow_vector)
        label = self.predict(pow_vector)
        confidence = 0.65

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))

            if hasattr(self.model, "classes_"):
                class_index = int(np.argmax(probabilities))
                label = _normalise_prediction_output(self.model.classes_[class_index])
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(features)
            score_array = np.asarray(scores)
            confidence = float(1.0 / (1.0 + np.exp(-np.max(score_array))))

        return {
            "label": label,
            "confidence": confidence,
        }


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
        if self.model is None:
            self.model = RandomForestClassifier(**self.hyperparams)
        self.model.fit(np.asarray(pow_vectors, dtype=float), np.asarray(labels))
        return self.model

    def predict(self, pow_vector: List[float]) -> str:
        if self.model is None:
            raise RuntimeError("Random Forest model is not loaded")

        features = self._prepare_features(pow_vector)
        prediction = self.model.predict(features)[0]
        return _normalise_prediction_output(prediction)

    def predict_with_confidence(self, pow_vector: List[float]) -> Dict:
        if self.model is None:
            raise RuntimeError("Random Forest model is not loaded")

        features = self._prepare_features(pow_vector)
        label = self.predict(pow_vector)
        confidence = 0.65

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))

            if hasattr(self.model, "classes_"):
                class_index = int(np.argmax(probabilities))
                label = _normalise_prediction_output(self.model.classes_[class_index])

        return {
            "label": label,
            "confidence": confidence,
        }


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

    def select(
        self,
        name: str,
        model_path: str = None,
        hyperparams: dict = None,
        num_classes: int = None,
        **kwargs,
    ):
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
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        **kwargs,
    ):
        impl = self._registry.get(name)
        if impl is None:
            raise ValueError(f"Classifier '{name}' not registered")

        self.active = impl(
            model_path=model_path if os.path.exists(model_path) else None,
            hyperparams=hyperparams,
            num_classes=num_classes,
            input_len=self.input_len,
            **kwargs,
        )
        if os.path.exists(model_path):
            self.active.validate_existing_model(model_path)

        stratify_labels = labels if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            pow_vectors,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )

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
