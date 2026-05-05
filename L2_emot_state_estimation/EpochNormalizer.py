import numpy as np


class EPOCCrossSessionNormalizer:
    def __init__(self):
        # Learned from DREAMER
        self.global_mu = None
        self.global_sigma = None

        # Per-session offset (critical for EPOC)
        self.session_baseline = None

    def fit_dreamer(self, dreamer_features):
        """
        DREAMER has multiple subjects, multiple sessions.
        Compute statistics that generalize across sessions.
        """
        # Option 1: Subject-independent (more generalizable)
        self.global_mu = np.mean(dreamer_features, axis=0)
        self.global_sigma = np.std(dreamer_features, axis=0)

        # Option 2: Subject-dependent with pooling (if you have subject IDs)
        # self.subject_stats = {sid: (mu, sigma) for sid in dreamer_subjects}

        return self

    def calibrate_session(self, epoch_baseline_features):
        """
        30-60 seconds of neutral/eyes-open at session start.
        EPOC sessions vary due to:
        - Electrode repositioning
        - Impedance differences
        - Skin moisture/saline concentration
        """
        self.session_baseline = np.median(epoch_baseline_features, axis=0)
        return self

    def transform(self, features, source="dreamer"):
        # Z-score with global parameters
        z = (features - self.global_mu) / (self.global_sigma + 1e-8)

        if source == "epoch" and self.session_baseline is not None:
            # Remove session-specific offset (common mode)
            z -= (self.session_baseline - self.global_mu) / self.global_sigma

        return z
