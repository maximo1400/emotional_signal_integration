"""
Valence-Arousal Classifier Selector / Manager

This module provides:
- `BaseClassifier` abstract interface for classifier implementations
- simple adapter skeletons for KNN/SVM/RandomForest
- `ClassifierManager` — registry + selector that instantiates the chosen classifier
- backward-compatible `VAClassifier` wrapper to preserve existing constructor usage
"""

import abc
import numpy as np
from typing import Dict, List, Type
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


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

    @abc.abstractmethod
    def fit(self, pow_vectors: List[List[float]], labels: List[str]):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, pow_vector: List[float]) -> Dict:
        raise NotImplementedError()

    def batch_predict(self, pow_vectors: List[List[float]]) -> List[Dict]:
        return [self.predict(v) for v in pow_vectors]

    def load_model(self, model_path: str = None):
        """Optional model loading hook for adapters."""
        return None

    def save_model(self, model_path: str):
        joblib.dump(
            {
                "model": self.model,
                "classifier": self.name,
                "hyperparams": getattr(self, "hyperparams", None),
                "num_classes": getattr(self, "num_classes", None),
            },
            model_path,
        )
        return model_path


# Adapter skeletons for external model wrappers
class KNNClassifierAdapter(BaseClassifier):
    name = "knn"

    def __init__(
        self,
        model_path: str = None,
        hyperparams: dict = None,
        num_classes: int = None,
    ):
        self.model = None
        self.hyperparams = hyperparams["knn"]
        self.model_path = model_path
        self.num_classes = num_classes
        if model_path:
            self.load_model(model_path)

    def fit(self, pow_vectors: List[List[float]], labels: List[str]):

        self.model = KNeighborsClassifier(**self.hyperparams)
        self.model.fit(np.asarray(pow_vectors, dtype=float), np.asarray(labels))
        return self.model

    def load_model(self, model_path: str = None):
        self.model_path = model_path or self.model_path

        loaded = joblib.load(self.model_path)
        self.model = (
            loaded.get("model")
            if isinstance(loaded, dict) and "model" in loaded
            else loaded
        )
        return self.model

    def predict(self, pow_vector: List[float]) -> Dict:
        if self.model is None:
            raise RuntimeError("KNN model is not loaded")

        features = np.asarray(pow_vector, dtype=float).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        label = _normalise_prediction_output(prediction)

        confidence = 0.65
        if hasattr(self.model, "predict_proba"):
            try:
                probabilities = self.model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
                if hasattr(self.model, "classes_"):
                    class_index = int(np.argmax(probabilities))
                    label = _normalise_prediction_output(
                        self.model.classes_[class_index]
                    )
            except Exception:
                pass

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
    ):
        self.model = None
        self.hyperparams = hyperparams["svm"]
        self.model_path = model_path
        self.num_classes = num_classes
        if model_path:
            self.load_model(model_path)

    def fit(self, pow_vectors: List[List[float]], labels: List[str]):

        fit_params = dict(self.hyperparams)
        fit_params.setdefault("probability", True)
        self.model = SVC(**fit_params)
        self.model.fit(np.asarray(pow_vectors, dtype=float), np.asarray(labels))
        return self.model

    def load_model(self, model_path: str = None):
        self.model_path = model_path or self.model_path
        if not self.model_path:
            return None

        loaded = joblib.load(self.model_path)
        self.model = (
            loaded.get("model")
            if isinstance(loaded, dict) and "model" in loaded
            else loaded
        )
        return self.model

    def predict(self, pow_vector: List[float]) -> Dict:
        if self.model is None:
            raise RuntimeError("SVM model is not loaded")

        features = np.asarray(pow_vector, dtype=float).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        label = _normalise_prediction_output(prediction)

        confidence = 0.65
        if hasattr(self.model, "decision_function"):
            try:
                scores = self.model.decision_function(features)
                score_array = np.asarray(scores)
                confidence = float(1.0 / (1.0 + np.exp(-np.max(score_array))))
            except Exception:
                pass
        elif hasattr(self.model, "predict_proba"):
            try:
                probabilities = self.model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
                if hasattr(self.model, "classes_"):
                    class_index = int(np.argmax(probabilities))
                    label = _normalise_prediction_output(
                        self.model.classes_[class_index]
                    )
            except Exception:
                pass

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
    ):
        self.model = None
        self.hyperparams = hyperparams["random_forest"]
        self.model_path = model_path
        self.num_classes = num_classes
        if model_path:
            self.load_model(model_path)

    def fit(self, pow_vectors: List[List[float]], labels: List[str]):

        self.model = RandomForestClassifier(**self.hyperparams)
        self.model.fit(np.asarray(pow_vectors, dtype=float), np.asarray(labels))
        return self.model

    def load_model(self, model_path: str = None):
        self.model_path = model_path or self.model_path
        if not self.model_path:
            return None

        loaded = joblib.load(self.model_path)
        self.model = (
            loaded.get("model")
            if isinstance(loaded, dict) and "model" in loaded
            else loaded
        )
        return self.model

    def predict(self, pow_vector: List[float]) -> Dict:
        if self.model is None:
            raise RuntimeError("Random Forest model is not loaded")

        features = np.asarray(pow_vector, dtype=float).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        label = _normalise_prediction_output(prediction)

        confidence = 0.65
        if hasattr(self.model, "predict_proba"):
            try:
                probabilities = self.model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
                if hasattr(self.model, "classes_"):
                    class_index = int(np.argmax(probabilities))
                    label = _normalise_prediction_output(
                        self.model.classes_[class_index]
                    )
            except Exception:
                pass

        return {
            "label": label,
            "confidence": confidence,
        }


class ClassifierManager:
    """Registry + selector for classifier implementations.

    Usage:
        mgr.register('va_stub', KNNClassifier)
        mgr.select('va_stub', **kwargs)
        out = mgr.predict(vec)
    """

    _registry: Dict[str, Type[BaseClassifier]] = {}

    def __init__(self):
        self.active: BaseClassifier = None

        # Register built-ins
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
            **kwargs,
        )
        # allow loading model if provided
        if model_path:
            try:
                self.active.load_model(model_path)
            except Exception:
                pass
        return self.active

    def train(
        self,
        name: str,
        pow_vectors: List[List[float]],
        labels: List[str],
        model_path: str = None,
        hyperparams: dict = None,
        num_classes: int = None,
        **kwargs,
    ):
        impl = self._registry.get(name)
        if impl is None:
            raise ValueError(f"Classifier '{name}' not registered")

        self.active = impl(
            model_path=None,
            hyperparams=hyperparams or {},
            num_classes=num_classes,
            **kwargs,
        )
        # TODO: un-hardcode train/test split and stratification strategy
        X_train, X_test, y_train, y_test = train_test_split(
            pow_vectors, labels, test_size=0.2, random_state=42, stratify=labels
        )
        self.active.fit(X_train, y_train)
        y_pred = self.active.predict(X_test)

        if model_path:
            self.active.save_model(model_path)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        return self.active

    def predict(self, pow_vector: List[float]) -> Dict:
        if not self.active:
            raise RuntimeError("No classifier selected")
        return self.active.predict(pow_vector)

    def batch_predict(self, pow_vectors: List[List[float]]) -> List[Dict]:
        if not self.active:
            raise RuntimeError("No classifier selected")
        return self.active.batch_predict(pow_vectors)


__all__ = ["BaseClassifier", "ClassifierManager"]
