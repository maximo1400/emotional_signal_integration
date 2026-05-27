import numpy as np
from config_loader import get_config

class EPOCCrossSessionNormalizer:
    def __init__(self, is_epoch_data: bool = True, calibration_time: int = 45):
        self.is_epoch_data = is_epoch_data
        
        # Load global statistics from config
        config = get_config(["global_mu", "global_sigma", "emotiv_pow_frec"])
        self.global_mu = np.array(config["global_mu"])
        self.global_sigma = np.array(config["global_sigma"])
        freq = config["emotiv_pow_frec"]

        self.session_baseline = None
        self.calibration_buffer = []
        self.calibration_done = False
        
        self.calibration_samples_required = calibration_time * freq

    def calibrate_session(self, pow_row):
        """
        30-60 seconds of neutral/eyes-open at session start.

        """
        if not self.is_epoch_data or self.calibration_done:
            return 
            
        self.calibration_buffer.append(pow_row)
        
        if len(self.calibration_buffer) >= self.calibration_samples_required:
            self.session_baseline = np.median(self.calibration_buffer, axis=0)
            self.calibration_done = True
            print("Session calibration complete. Baseline computed.")
            
        return

    def transform(self, pow_row, eps = 1e-10):
        if not self.is_epoch_data:
            return pow_row

        # Z-score with global parameters
        z = (pow_row - self.global_mu) / (self.global_sigma + eps)

        if self.session_baseline is not None:
            # Remove session-specific offset (common mode)
            z -= (self.session_baseline - self.global_mu) / (self.global_sigma + eps)

        return z

    def new_row(self, pow_row):
        self.calibrate_session(pow_row)
        return self.transform(pow_row)