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

        freq: float = config["emotiv_pow_frec"]
        self.calibration_len: int = int(config["calibration_time"] * freq)

        simple_calibration: bool = config["simple_calibration"]
        if simple_calibration:
            self.calibration_done = True

    def _finalize_calibration(self) -> None:
        """Computes the median baseline from the buffer and sets calibration_done."""
        self.mu = np.median(self.calibration_buffer, axis=0)
        self.calibration_done = True
        self.calibration_buffer = []  # Free memory
        print("Session calibration complete. Baseline computed.")

    def _transform(self, pow_arr: np.ndarray) -> np.ndarray:
        """Applies Z-score normalization using current mu and sigma."""
        return (pow_arr - self.mu) / (self.global_sigma + 1e-10)

    def new_row(self, pow_row: list[float]) -> list[float]:
        if not self.normalizer_enabled:
            return pow_row

        pow_row_arr = np.array(pow_row)
        self.calibration_buffer.append(pow_row_arr)
        z = self._transform(pow_row_arr)

        # gets new calibration state if enough samples have been collected
        if (
            not self.calibration_done
            and len(self.calibration_buffer) >= self.calibration_len
        ):
            self._finalize_calibration()

        return z.tolist()

    def process_batch(self, pow_rows: np.ndarray) -> list[list[float]]:
        if not self.normalizer_enabled:
            return pow_rows.tolist()

        if self.calibration_done:
            return self._transform(pow_rows).tolist()

        samples_needed = self.calibration_len - len(self.calibration_buffer)

        if len(pow_rows) < samples_needed:
            # Not enough samples to finish calibration in this batch
            self.calibration_buffer.extend(pow_rows)
            return self._transform(pow_rows).tolist()

        # We have enough samples to finish calibration mid-batch!
        # 1. Normalize the calibrating part with the global_mu
        calib_part = pow_rows[:samples_needed]
        z_calib = self._transform(calib_part)

        # 2. Finalize calibration state
        self.calibration_buffer.extend(calib_part)
        self._finalize_calibration()

        # 3. Normalize the remaining part with the newly computed session_mu
        remaining_part = pow_rows[samples_needed:]
        if len(remaining_part) > 0:
            z_rem = self._transform(remaining_part)
            return z_calib.tolist() + z_rem.tolist()

        return z_calib.tolist()
