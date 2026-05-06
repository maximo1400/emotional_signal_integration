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

    Input: power vector (floats) in order of POW_COLUMNS
    Output: {valence, arousal, label, confidence}
    """

    # POW_COLUMNS from L1 (must match exactly)
    POW_COLUMNS = [
        "AF3/theta",
        "AF3/alpha",
        "AF3/betaL",
        "AF3/betaH",
        "AF3/gamma",
        "F7/theta",
        "F7/alpha",
        "F7/betaL",
        "F7/betaH",
        "F7/gamma",
        "F3/theta",
        "F3/alpha",
        "F3/betaL",
        "F3/betaH",
        "F3/gamma",
        "FC5/theta",
        "FC5/alpha",
        "FC5/betaL",
        "FC5/betaH",
        "FC5/gamma",
        "T7/theta",
        "T7/alpha",
        "T7/betaL",
        "T7/betaH",
        "T7/gamma",
        "P7/theta",
        "P7/alpha",
        "P7/betaL",
        "P7/betaH",
        "P7/gamma",
        "O1/theta",
        "O1/alpha",
        "O1/betaL",
        "O1/betaH",
        "O1/gamma",
        "O2/theta",
        "O2/alpha",
        "O2/betaL",
        "O2/betaH",
        "O2/gamma",
        "P8/theta",
        "P8/alpha",
        "P8/betaL",
        "P8/betaH",
        "P8/gamma",
        "T8/theta",
        "T8/alpha",
        "T8/betaL",
        "T8/betaH",
        "T8/gamma",
        "FC6/theta",
        "FC6/alpha",
        "FC6/betaL",
        "FC6/betaH",
        "FC6/gamma",
        "F4/theta",
        "F4/alpha",
        "F4/betaL",
        "F4/betaH",
        "F4/gamma",
        "F8/theta",
        "F8/alpha",
        "F8/betaL",
        "F8/betaH",
        "F8/gamma",
        "AF4/theta",
        "AF4/alpha",
        "AF4/betaL",
        "AF4/betaH",
        "AF4/gamma",
    ]

    # Emotion ranges for VA space
    VA_RANGES = {
        "bored": {"valence": (-1.0, -0.3), "arousal": (-1.0, -0.3)},
        "frustrated": {"valence": (-1.0, -0.3), "arousal": (-0.3, 0.3)},
        "angry": {"valence": (-1.0, -0.3), "arousal": (0.3, 1.0)},
        "tired": {"valence": (-0.3, 0.3), "arousal": (-1.0, -0.3)},
        "neutral": {"valence": (-0.3, 0.3), "arousal": (-0.3, 0.3)},
        "surprise": {"valence": (-0.3, 0.3), "arousal": (0.3, 1.0)},
        "calm": {"valence": (0.3, 1.0), "arousal": (-1.0, -0.3)},
        "happy": {"valence": (0.3, 1.0), "arousal": (-0.3, 0.3)},
        "excited": {"valence": (0.3, 1.0), "arousal": (0.3, 1.0)},
    }

    def __init__(self, model_path: str = None):
        """
        Initialize classifier.

        Args:
            model_path: Path to a trained model (optional, not used in stub)
        """
        self.model_path = model_path
        self.model = None
        print("VAClassifier initialized (stub mode).")

    def predict(self, pow_vector: List[float]) -> Dict:
        """
        Predict valence and arousal from a power vector.

        Args:
            pow_vector: List of power values (70 features) in POW_COLUMNS order

        Returns:
            Dict with keys: valence, arousal, label, confidence, timestamp
        """
        if len(pow_vector) != len(self.POW_COLUMNS):
            raise ValueError(
                f"Expected {len(self.POW_COLUMNS)} power values, got {len(pow_vector)}"
            )

        # STUB: Simple heuristic classifier
        # Replace this with a trained model (sklearn, PyTorch, etc.)

        pow_array = np.array(pow_vector)

        # Example: use band ratios as features
        # Alpha / (Beta + Theta) -> arousal proxy
        # Frontal alpha asymmetry -> valence proxy

        alpha_indices = [i for i, col in enumerate(self.POW_COLUMNS) if "alpha" in col]
        beta_indices = [
            i
            for i, col in enumerate(self.POW_COLUMNS)
            if "betaL" in col or "betaH" in col
        ]
        theta_indices = [i for i, col in enumerate(self.POW_COLUMNS) if "theta" in col]

        alpha_mean = np.mean(pow_array[alpha_indices]) if alpha_indices else 0.0
        beta_mean = np.mean(pow_array[beta_indices]) if beta_indices else 0.0
        theta_mean = np.mean(pow_array[theta_indices]) if theta_indices else 0.0

        # Normalize to [-1, 1]
        arousal = self._normalize_to_va(alpha_mean / (beta_mean + theta_mean + 1e-6))

        # Simple frontal asymmetry: left vs right frontal regions
        left_frontal = [
            i for i, col in enumerate(self.POW_COLUMNS) if "F3" in col or "AF3" in col
        ]
        right_frontal = [
            i for i, col in enumerate(self.POW_COLUMNS) if "F4" in col or "AF4" in col
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

        for label, ranges in self.VA_RANGES.items():
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
