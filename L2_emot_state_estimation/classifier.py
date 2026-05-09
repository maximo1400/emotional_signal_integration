"""
Valence-Arousal Classifier Stub

Skeleton classifier that consumes power vectors and outputs
valence/arousal estimates. Can be replaced with a trained model.
"""

import numpy as np
from typing import Dict, List


class VAClassifier:
    """
    Stub valence-arousal classifier.

    Input: power vector (floats) in order of pow_columns
    Output: {valence, arousal, label, confidence}
    """

    def __init__(
        self, pow_columns: list, emotional_states_areas: list, model_path: str = None
    ):
        """
        Initialize classifier.

        Args:
            model_path: Path to a trained model (optional, not used in stub)
        """
        self.model_path = model_path
        self.model = None
        self.pow_columns = pow_columns
        self.emot_states_areas = self._reshape_va_ranges(emotional_states_areas)
        print("VAClassifier initialized (stub mode).")

    def _reshape_va_ranges(self, emot_states_areas) -> Dict:
        """
        Reshape emotional_states_areas for ease of use.
        Args:
            emotional_states_areas: Input data (dict, list, or other format)

        Returns:
            Dict: Converted format to:
        { "emotion_label": {"valence": (min, max),"arousal": (min, max)}...}
        """

        reshaped = {}
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

    def predict(self, pow_vector: List[float]) -> Dict:
        """
        Predict valence and arousal from a power vector.

        Args:
            pow_vector: List of power values (70 features) in pow_columns order

        Returns:
            Dict with keys: valence, arousal, label, confidence, timestamp
        """

        # STUB: Simple heuristic classifier
        # Replace this with a trained model (sklearn, PyTorch, etc.)

        pow_array = np.array(pow_vector)

        # Example: use band ratios as features
        # Alpha / (Beta + Theta) -> arousal proxy
        # Frontal alpha asymmetry -> valence proxy

        alpha_indices = [i for i, col in enumerate(self.pow_columns) if "alpha" in col]
        beta_indices = [
            i
            for i, col in enumerate(self.pow_columns)
            if "betaL" in col or "betaH" in col
        ]
        theta_indices = [i for i, col in enumerate(self.pow_columns) if "theta" in col]

        alpha_mean = np.mean(pow_array[alpha_indices]) if alpha_indices else 0.0
        beta_mean = np.mean(pow_array[beta_indices]) if beta_indices else 0.0
        theta_mean = np.mean(pow_array[theta_indices]) if theta_indices else 0.0

        # Normalize to [-1, 1]
        arousal = self._normalize_to_va(alpha_mean / (beta_mean + theta_mean + 1e-6))

        # Simple frontal asymmetry: left vs right frontal regions
        left_frontal = [
            i for i, col in enumerate(self.pow_columns) if "F3" in col or "AF3" in col
        ]
        right_frontal = [
            i for i, col in enumerate(self.pow_columns) if "F4" in col or "AF4" in col
        ]

        left_power = np.mean(pow_array[left_frontal]) if left_frontal else 0.0
        right_power = np.mean(pow_array[right_frontal]) if right_frontal else 0.0

        valence = self._normalize_to_va(right_power - left_power)

        # Map to emotion label
        label = self._map_to_label(valence, arousal)

        # Stub confidence (in real model, use model output probabilities)
        confidence = 0.65

        return {
            "valence": valence,
            "arousal": arousal,
            "label": label,
            "confidence": confidence,
        }

    def _normalize_to_va(self, value: float, scale: float = 1.0) -> float:
        """Normalize a raw feature to [-1, 1] VA range."""
        return float(np.tanh(value * scale))

    def _map_to_label(self, valence: float, arousal: float) -> str:
        """Map (valence, arousal) to closest emotion label."""
        min_dist = float("inf")
        best_label = "neutral"

        for label, ranges in self.emot_states_areas.items():
            val_mid = (ranges["valence"][0] + ranges["valence"][1]) / 2
            ar_mid = (ranges["arousal"][0] + ranges["arousal"][1]) / 2

            dist = (valence - val_mid) ** 2 + (arousal - ar_mid) ** 2
            if dist < min_dist:
                min_dist = dist
                best_label = label

        return best_label

    def batch_predict(self, pow_vectors: List[List[float]]) -> List[Dict]:
        """Predict on a batch of power vectors."""
        return [self.predict(vec) for vec in pow_vectors]
