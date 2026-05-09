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
from typing import Dict, List, Optional, Type
import joblib


def _reshape_va_ranges(emot_states_areas) -> Dict:
    """
    Convert input emotional state areas into the canonical mapping:
    { "label": {"valence": (min, max), "arousal": (min, max)} }

    """

    reshaped = {}
    # list-of-dicts format
    for item in emot_states_areas:
        label = item["label"]
        # id = item["id"]
        val_min = item["valence_min"]
        val_max = item["valence_max"]
        ar_min = item["arousal_min"]
        ar_max = item["arousal_max"]
        reshaped[label] = {
            "valence": (val_min, val_max),
            "arousal": (ar_min, ar_max),
        }
    return reshaped


def _label_to_va(label: str, emot_states_areas: Dict) -> Dict:
    """Convert an emotion label into a representative VA point."""
    ranges = emot_states_areas.get(label)

    valence = ranges["valence"]
    arousal = ranges["arousal"]
    return {
        "valence": float((valence[0] + valence[1]) / 2),
        "arousal": float((arousal[0] + arousal[1]) / 2),
    }


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
    def predict(self, pow_vector: List[float]) -> Dict:
        raise NotImplementedError()

    def batch_predict(self, pow_vectors: List[List[float]]) -> List[Dict]:
        return [self.predict(v) for v in pow_vectors]

    def load_model(self, model_path: Optional[str] = None):
        """Optional model loading hook for adapters."""
        return None


# Adapter skeletons for external model wrappers
class KNNClassifierAdapter(BaseClassifier):
    name = "knn"

    def __init__(
        self,
        pow_columns: list,
        emot_states_areas: list,
        model_path: Optional[str] = None,
        hyperparams: Optional[dict] = None,
    ):
        self.pow_columns = pow_columns
        self.emot_states_areas = _reshape_va_ranges(emot_states_areas)
        self.model = None
        self.hyperparams = hyperparams or {}
        self.model_path = model_path
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: Optional[str] = None):
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

        va = _label_to_va(label, self.emot_states_areas)
        return {
            "valence": va["valence"],
            "arousal": va["arousal"],
            "label": label,
            "confidence": confidence,
        }


class SVMClassifierAdapter(BaseClassifier):
    name = "svm"

    def __init__(
        self,
        pow_columns: list,
        emot_states_areas: list,
        model_path: Optional[str] = None,
        hyperparams: Optional[dict] = None,
    ):
        self.pow_columns = pow_columns
        self.emot_states_areas = _reshape_va_ranges(emot_states_areas)
        self.model = None
        self.hyperparams = hyperparams or {}
        self.model_path = model_path
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: Optional[str] = None):
        self.model_path = model_path or self.model_path
        if not self.model_path:
            return None
        if joblib is None:
            raise ImportError("joblib is required to load sklearn models")

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

        va = _label_to_va(label, self.emot_states_areas)
        return {
            "valence": va["valence"],
            "arousal": va["arousal"],
            "label": label,
            "confidence": confidence,
        }


class RFClassifierAdapter(BaseClassifier):
    name = "random_forest"

    def __init__(
        self,
        pow_columns: list,
        emot_states_areas: list,
        model_path: Optional[str] = None,
        hyperparams: Optional[dict] = None,
    ):
        self.pow_columns = pow_columns
        self.emot_states_areas = _reshape_va_ranges(emot_states_areas)
        self.model = None
        self.hyperparams = hyperparams or {}
        self.model_path = model_path
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: Optional[str] = None):
        self.model_path = model_path or self.model_path
        if not self.model_path:
            return None
        if joblib is None:
            raise ImportError("joblib is required to load sklearn models")

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

        va = _label_to_va(label, self.emot_states_areas)
        return {
            "valence": va["valence"],
            "arousal": va["arousal"],
            "label": label,
            "confidence": confidence,
        }


class ClassifierManager:
    """Registry + selector for classifier implementations.

    Usage:
        mgr = ClassifierManager(pow_columns, emot_states_areas)
        mgr.register('va_stub', KNNClassifier)
        mgr.select('va_stub', **kwargs)
        out = mgr.predict(vec)
    """

    _registry: Dict[str, Type[BaseClassifier]] = {}

    def __init__(self, pow_columns: list, emot_states_areas: list):
        self.pow_columns = pow_columns
        self.emot_states_areas = emot_states_areas
        self.active: Optional[BaseClassifier] = None

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
        model_path: Optional[str] = None,
        hyperparams: Optional[dict] = None,
        **kwargs,
    ):
        impl = self._registry.get(name)
        if impl is None:
            raise ValueError(f"Classifier '{name}' not registered")

        self.active = impl(
            self.pow_columns,
            self.emot_states_areas,
            model_path=model_path,
            hyperparams=hyperparams or {},
            **kwargs,
        )
        # allow loading model if provided
        if model_path:
            try:
                self.active.load_model(model_path)
            except Exception:
                pass
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
