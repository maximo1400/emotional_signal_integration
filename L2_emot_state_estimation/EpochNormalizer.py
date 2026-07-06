import numpy as np

from config_loader import get_config


class EPOCCrossSessionNormalizer:
    def __init__(self, is_epoch_data: bool = True) -> None:
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
        self.global_mu: np.ndarray = np.array(config["global_mu"])
        self.global_sigma: np.ndarray = np.array(config["global_sigma"])
        self.normalize_enabled: bool = is_epoch_data and config["enable_normalizer"]
        self.simple_calibration: bool = config["simple_calibration"]
        freq: float = config["emotiv_pow_frec"]
        self.calibration_samples_required: int = int(config["calibration_time"] * freq)

    def _calibrate_session(self, pow_row: list[float]) -> None:
        if self.simple_calibration or self.calibration_done:
            return

        self.calibration_buffer.append(np.array(pow_row))

        if len(self.calibration_buffer) >= self.calibration_samples_required:
            self.session_baseline = np.median(self.calibration_buffer, axis=0)
            self.calibration_done = True
            self.calibration_buffer = []  # Clear buffer
            print("Session calibration complete. Baseline computed.")

    def _transform(self, pow_row: list[float], eps: float = 1e-10) -> list[float]:
        pow_row_arr = np.array(pow_row)

        # Use session baseline if calibrated, otherwise fallback to global_mu
        baseline = self.session_baseline if self.calibration_done else self.global_mu

        z: np.ndarray = (pow_row_arr - baseline) / (self.global_sigma + eps)

        return z.tolist()

    def new_row(self, pow_row: list[float]) -> list[float]:
        if not self.normalize_enabled:
            return pow_row

        self._calibrate_session(pow_row)
        return self._transform(pow_row)

    def process_batch(self, pow_rows: np.ndarray) -> list[list[float]]:
        if not self.normalize_enabled:
            return pow_rows.tolist()

        normalized_rows = []
        for row in pow_rows:
            row = row.tolist()
            self._calibrate_session(row)
            normalized_rows.append(self._transform(row))
        return normalized_rows
