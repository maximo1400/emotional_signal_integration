import sys
from pathlib import Path
import time
from typing import List

# Add parent directory to path to import config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import get_config
import numpy as np


class FeatureSelector:
    def __init__(
        self,
        features=["l2_pow_columns", "features_to_add", "asymmetries"],
    ):
        self.pow_columns = []
        self.pow_columns_mask = []
        self.features_to_add = []
        self.sensor_info = {}
        self.col_index = {}
        self.asymmetries = []
        self.load_config(features)
        self.labels = self.get_final_feature_names()
        self.pow = []

    def load_config(self, config_keys):
        epoch_data = get_config(["POW_COLUMNS", "epoch_sensors"])
        self.pow_columns = epoch_data["POW_COLUMNS"]
        self.sensor_info = epoch_data["epoch_sensors"]
        self.col_index = {col: i for i, col in enumerate(self.pow_columns)}

        feat = get_config(config_keys)
        self.features_to_add = feat["features_to_add"]
        self.asymmetries = feat["asymmetries"]
        self.pow_columns_mask = [True] * len(self.pow_columns)
        l2_pow_columns = feat["l2_pow_columns"]
        if len(l2_pow_columns) > 0:
            self.pow_columns_mask = [col in l2_pow_columns for col in self.pow_columns]

    def process_data(self, pow_data: List[float]) -> List[float]:
        data = self.filter_pow_columns(pow_data)
        data.extend(self.add_features(pow_data))
        row = data + [time.time()]
        self.pow.append(row)
        return data

    def get_final_feature_names(self) -> List[str]:
        feature_names = [
            col for col, keep in zip(self.pow_columns, self.pow_columns_mask) if keep
        ]

        for area, type, band in self.asymmetries:
            feature_names.append(f"{area}_{type}_{band}_asymmetry")

        for feat in self.features_to_add:
            feature_names.append(feat)

        feature_names.append("timestamp")
        return feature_names

    def filter_pow_columns(self, pow_data: List[float]) -> List[float]:
        return [val for val, keep in zip(pow_data, self.pow_columns_mask) if keep]

    def add_features(self, pow_data: List[float]) -> List[float]:
        features = []
        for area, type, band in self.asymmetries:
            # print(f"Adding asymmetry feature: {area}_{type}_{band}")
            features.append(self.calc_asymmetry(pow_data, area, type, band))

        for feat in self.features_to_add:
            # print(f"Adding feature: {feat}")
            pass

        return features

    def calc_asymmetry(
        self,
        pow_data: List[float],
        area: str,
        diff_method: str,
        frec_band: str,
        eps: float = 1e-10,
    ) -> float:

        if area == "frontal":
            pairs = self.sensor_info["frontal_pairs"]
        elif area == "parietal":
            pairs = self.sensor_info["parietal_pairs"]
        asym = []

        for left, right in pairs:
            # print(f"Calculating {area} asymmetry for pair: {left} - {right}")
            left_col = f"{left}/{frec_band}"
            right_col = f"{right}/{frec_band}"

            left_idx = self.col_index.get(left_col)
            right_idx = self.col_index.get(right_col)

            # Raw values with epsilon for numerical stability
            left_raw = pow_data[left_idx] + eps
            right_raw = pow_data[right_idx] + eps

            if diff_method == "ratio":
                asym.append(-(right_raw / left_raw))
            else:  # diff_method == "difference"
                asym.append(-((right_raw) - (left_raw)))

        return np.mean(asym)

    def calc_avg_frontal_beta(self, pow_data: List[float]) -> float:
        beta_values = []
        for electrode in self.sensor_info["frontal_electrodes"]:
            for band in ["betaL", "betaH"]:
                col_name = f"{electrode}/{band}"
                idx = self.col_index.get(col_name)
                beta_values.append(pow_data[idx])
        return np.mean(beta_values)
