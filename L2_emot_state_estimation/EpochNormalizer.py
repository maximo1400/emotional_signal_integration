import numpy as np

from config_loader import get_config


class EPOCCrossSessionNormalizer:
    def __init__(self) -> None:
        self.calibration_done: bool = False
        self.calibration_buffer: list[np.ndarray] = []
        # self.session_baseline: np.ndarray = np.array([])

        config_keys = [
            "global_mu",
            "global_sigma",
            "emotiv_pow_frec",
            "enable_normalizer",
            "calibration_time",
            "simple_calibration",
        ]
        config = get_config(config_keys)
        self.mu: np.ndarray = np.array(config["global_mu"])
        self.global_sigma: np.ndarray = np.array(config["global_sigma"])
        self.normalizer_enabled: bool = config["enable_normalizer"]
        self.simple_calibration: bool = config["simple_calibration"]
        freq: float = config["emotiv_pow_frec"]
        self.calibration_samples_required: int = int(config["calibration_time"] * freq)

    def _calibrate_session(self, pow_row: list[float]) -> None:
        if self.simple_calibration or self.calibration_done:
            return

        self.calibration_buffer.append(np.array(pow_row))

        if len(self.calibration_buffer) >= self.calibration_samples_required:
            self.mu = np.median(self.calibration_buffer, axis=0)
            self.calibration_done = True
            self.calibration_buffer = []  # Clear buffer
            print("Session calibration complete. Baseline computed.")

    def _transform(self, pow_row: list[float], eps: float = 1e-10) -> list[float]:
        pow_row_arr = np.array(pow_row)
        z: np.ndarray = (pow_row_arr - self.mu) / (self.global_sigma + eps)
        return z.tolist()

    def new_row(self, pow_row: list[float]) -> list[float]:
        if not self.normalizer_enabled:
            return pow_row

        self._calibrate_session(pow_row)
        return self._transform(pow_row)

    def process_batch(self, pow_rows: np.ndarray) -> list[list[float]]:
        if not self.normalizer_enabled:
            return pow_rows.tolist()

        normalized_rows = []
        for row in pow_rows:
            row = row.tolist()
            self._calibrate_session(row)
            normalized_rows.append(self._transform(row))
        return normalized_rows
