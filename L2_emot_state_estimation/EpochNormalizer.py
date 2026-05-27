import numpy as np
from config_loader import get_config


class EPOCCrossSessionNormalizer:
    def __init__(self, is_epoch_data: bool = True, calibration_time: int = 45) -> None:
        self.is_epoch_data: bool = is_epoch_data

        # Load global statistics from config
        config = get_config(["global_mu", "global_sigma", "emotiv_pow_frec"])
        self.global_mu: np.ndarray = np.array(config["global_mu"])
        self.global_sigma: np.ndarray = np.array(config["global_sigma"])
        freq: float = config["emotiv_pow_frec"]

        self.session_baseline: np.ndarray = np.array([])
        self.calibration_buffer: list[np.ndarray] = []
        self.calibration_done: bool = False

        self.calibration_samples_required: int = int(calibration_time * freq)

    def calibrate_session(self, pow_row: np.ndarray) -> None:
        """
        30-60 seconds of neutral/eyes-open at session start.

        """
        if not self.is_epoch_data or self.calibration_done:
            return

        self.calibration_buffer.append(pow_row)

        if len(self.calibration_buffer) >= self.calibration_samples_required:
            self.session_baseline = np.median(self.calibration_buffer, axis=0)
            self.calibration_done = True
            self.calibration_buffer = []  # Clear buffer
            print("Session calibration complete. Baseline computed.")

    def transform(self, pow_row: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        if not self.is_epoch_data:
            return pow_row

        # Z-score with global parameters
        z: np.ndarray = (pow_row - self.global_mu) / (self.global_sigma + eps)

        if self.session_baseline is not None:
            # Remove session-specific offset (common mode)
            z -= (self.session_baseline - self.global_mu) / (self.global_sigma + eps)

        return z

    def new_row(self, pow_row: np.ndarray) -> np.ndarray:
        self.calibrate_session(pow_row)
        return self.transform(pow_row)
