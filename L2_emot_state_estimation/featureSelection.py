import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

# Add parent directory to path to import config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config


class FeatureSelector:
    """
    Feature selector / feature expander for Emotiv band-power vectors.

    Input pow_data must be length 70 and ordered according to POW_COLUMNS:

    Features from Garg et al. Chapter 2.3 that can be approximated from
    power-only data:

    - Band power aggregates
    - Shannon entropy over powers
    - Sub-band information quantity proxy
    - Hjorth mobility proxy from spectral moments
    - Hjorth complexity proxy from spectral moments
    - Differential entropy proxy
    - Differential asymmetry proxy
    - Rational asymmetry proxy
    - Median frequency proxy
    - Signal standard deviation proxy
    - Diffuse slowing proxy
    """

    def __init__(self):
        self.pow_columns = []
        self.pow_columns_mask = []
        self.features_to_add = []
        self.sensor_info = {}
        self.asymmetries = []

        self.band_frec_centrers = {}
        self.band_indices = {}
        self.sensor_band_idx = {}
        self.eps = 1e-10

        self.load_config()
        self.labels = self.get_final_feature_names()

    def load_config(self):
        features = ["l2_pow_columns", "features_to_add", "asymmetries"]
        keys = features + ["POW_COLUMNS", "epoch_sensors"]
        epoch_data = get_config(keys)
        self.pow_columns = epoch_data["POW_COLUMNS"]
        self.sensor_info = epoch_data["epoch_sensors"]
        self.features_to_add = epoch_data["features_to_add"]
        self.asymmetries = epoch_data["asymmetries"]
        l2_pow_columns = epoch_data["l2_pow_columns"]

        col_index = {col: i for i, col in enumerate(self.pow_columns)}

        self.pow_columns_mask = [True] * len(self.pow_columns)
        if len(l2_pow_columns) > 0:
            self.pow_columns_mask = [col in l2_pow_columns for col in self.pow_columns]

        epoch_bands = self.sensor_info["frequency_bands"]
        epoch_sensors = self.sensor_info["electrodes"]
        band_ranges = self.sensor_info["band_ranges"]

        for band in epoch_bands:
            # Compute band center frequencies for use in feature calculations
            low, high = band_ranges[band]
            center = (low + high) / 2.0
            self.band_frec_centrers[band] = center

            self.band_indices[band] = []
            for sensor in epoch_sensors:
                col = f"{sensor}/{band}"
                idx = col_index[col]
                # Get column indices for this band across all sensors
                self.band_indices[band].append(idx)
                # Store sensor-band to column index mapping for quick lookup in features
                self.sensor_band_idx[(sensor, band)] = idx

    def process_data(self, pow_data: List[float]) -> List[float]:
        """Selects and adds features to the input power data according to config."""
        if len(pow_data) != len(self.pow_columns):
            raise ValueError(
                f"Expected pow_data length {len(self.pow_columns)}, "
                f"got {len(pow_data)}."
            )

        pow_columns_mask_np = np.asarray(self.pow_columns_mask, dtype=bool)
        pow_arr = np.asarray(pow_data, dtype=float)
        pow_arr = np.nan_to_num(pow_arr, nan=0.0, posinf=0.0, neginf=0.0)
        pow_arr = np.maximum(pow_arr, self.eps)

        features = self.add_features(pow_arr)
        filtered_pow = pow_arr[pow_columns_mask_np].tolist()
        return filtered_pow + features

    def get_final_feature_names(self) -> List[str]:
        """Get the names and order of [pow_col, asym, features, timestamp] after selection and addition."""
        feature_names = [
            col for col, keep in zip(self.pow_columns, self.pow_columns_mask) if keep
        ]

        for area, asym_type, band in self.asymmetries:
            feature_names.append(f"{area}_{asym_type}_{band}_asymmetry")

        for feat in self.features_to_add:
            feature_names.append(feat)

        return feature_names

    def add_features(self, pow_data: np.ndarray) -> List[float]:
        features = []

        for area, asym_type, band in self.asymmetries:
            features.append(self.calc_asymmetry(pow_data, area, asym_type, band))

        for feat in self.features_to_add:
            features.append(self.calc_feature(pow_data, feat))

        return features

    def calc_feature(self, pow_data: np.ndarray, feat: str) -> float:
        name = feat.strip()
        lname = name.lower()

        if lname == "total_power":
            return self.calc_total_power(pow_data)

        if lname == "log_total_power":
            return float(np.log(self.calc_total_power(pow_data) + self.eps))

        if lname == "power_mean":
            return float(np.mean(pow_data))

        if lname == "power_std":
            return float(np.std(pow_data))

        if lname == "signal_std":
            return self.calc_signal_std_proxy(pow_data)

        if lname in ["shannon_entropy", "spectral_entropy"]:
            return self.calc_spectral_entropy(pow_data)

        if lname == "mean_sensor_spectral_entropy":
            return self.calc_mean_sensor_spectral_entropy(pow_data)

        if lname == "hjorth_mobility":
            return self.calc_hjorth_mobility_proxy(pow_data)

        if lname == "hjorth_complexity":
            return self.calc_hjorth_complexity_proxy(pow_data)

        if lname == "median_frequency":
            return self.calc_median_frequency_proxy(pow_data)

        if lname == "diffuse_slowing":
            return self.calc_diffuse_slowing_proxy(pow_data)

        if lname == "engagement_index":
            return self.calc_ratio(pow_data, ["betaL", "betaH"], ["alpha", "theta"])

        if lname == "beta_alpha_ratio":
            return self.calc_ratio(pow_data, ["betaL", "betaH"], ["alpha"])

        if lname == "theta_beta_ratio":
            return self.calc_ratio(pow_data, ["theta"], ["betaL", "betaH"])

        if lname == "theta_alpha_ratio":
            return self.calc_ratio(pow_data, ["theta"], ["alpha"])

        if lname == "gamma_beta_ratio":
            return self.calc_ratio(pow_data, ["gamma"], ["betaL", "betaH"])

        if lname == "slow_fast_ratio":
            return self.calc_ratio(
                pow_data, ["theta"], ["alpha", "betaL", "betaH", "gamma"]
            )

        if lname == "avg_frontal_beta":
            return self.calc_avg_frontal_beta(pow_data)

        if lname.startswith("mean_"):
            band = name[len("mean_") :]
            return self.calc_mean_band_power(pow_data, band)

        if lname.startswith("relative_"):
            band = name[len("relative_") :]
            return self.calc_relative_band_power(pow_data, band)

        if lname.startswith("siq_"):
            band = name[len("siq_") :]
            return self.calc_siq_proxy(pow_data, band)

        if lname.startswith("de_") and lname.endswith("_mean"):
            band = name[len("de_") : -len("_mean")]
            return self.calc_mean_differential_entropy_proxy(pow_data, band)

        raise ValueError(
            f"Unknown feature '{feat}'. Add it to calc_feature() or remove it "
            "from features_to_add."
        )

    def calc_asymmetry(
        self,
        pow_data: np.ndarray,
        area: str,
        diff_method: str,
        frec_band: str,
    ) -> float:
        """
        Calculates asymmetry over configured left/right electrode pairs.
        With your config from config_loader
        Supported diff_method values:

            - difference: mean(left - right)
            - ratio: mean(left / right)
            - log_ratio: mean(log(left) - log(right))
            - dasm (Differential asymmetry approximation):
                mean(DE(left) - DE(right))
            - rasm (Rational asymmetry approximation):
                mean(DE(left) / DE(right))

        DE is approximated from power as:sensor_info

            DE ~= 0.5 * log(2 * pi * e * power)
        """
        if area == "frontal":
            pairs = self.sensor_info["frontal_pairs"]
        elif area == "parietal":
            pairs = self.sensor_info["parietal_pairs"]
        elif area == "all":
            pairs = self.sensor_info["all_pairs"]
        else:
            print(f"Unknown area '{area}' for asymmetry calculation. Returning NaN.")
            return np.nan

        values = []

        for left, right in pairs:
            left_power = self.get_sensor_band_power(pow_data, left, frec_band)
            right_power = self.get_sensor_band_power(pow_data, right, frec_band)

            left_power = max(float(left_power), self.eps)
            right_power = max(float(right_power), self.eps)

            method = diff_method.lower()

            if method == "difference":
                values.append(left_power - right_power)

            elif method == "ratio":
                values.append(left_power / right_power)

            elif method == "log_ratio":
                values.append(np.log(left_power) - np.log(right_power))

            elif method in ["dasm", "de_difference"]:
                left_de = self.differential_entropy_proxy(left_power)
                right_de = self.differential_entropy_proxy(right_power)
                values.append(left_de - right_de)

            elif method in ["rasm", "de_ratio"]:
                left_de = self.differential_entropy_proxy(left_power)
                right_de = self.differential_entropy_proxy(right_power)
                values.append(left_de / (right_de + self.eps))

            else:
                raise ValueError(f"Unknown asymmetry method: {diff_method}")

        if len(values) == 0:
            return np.nan

        return float(np.mean(values))

    def calc_avg_frontal_beta(self, pow_data: np.ndarray) -> float:
        """Average beta power across frontal electrodes."""
        beta_values = []

        for electrode in self.sensor_info["frontal_electrodes"]:
            for band in ["betaL", "betaH"]:
                pow_value = self.get_sensor_band_power(pow_data, electrode, band)
                beta_values.append(pow_value)

        return float(np.mean(beta_values))

    def calc_total_power(self, pow_data: np.ndarray) -> float:
        """Sum of all band powers."""
        return float(np.sum(pow_data))

    def calc_signal_std_proxy(self, pow_data: np.ndarray) -> float:
        """
        Approximation.

        If summed band power approximates signal variance:

            std ~= sqrt(total_power)
        """
        return float(np.sqrt(self.calc_total_power(pow_data)))

    def calc_spectral_entropy(self, pow_data: np.ndarray) -> float:
        """
        Shannon entropy over the full 70-dimensional power vector.

        Normalized to [0, 1].
        """
        if len(pow_data) == 0:
            return np.nan

        probs = pow_data / (np.sum(pow_data) + self.eps)

        entropy = -np.sum(probs * np.log(probs + self.eps))
        max_entropy = np.log(len(probs))

        return float(entropy / (max_entropy + self.eps))

    def calc_mean_sensor_spectral_entropy(self, pow_data: np.ndarray) -> float:
        """For each sensor, calculate entropy over its 5 bands, then average."""
        entropies = []
        for sensor in self.sensor_info["electrodes"]:
            vals = []
            for band in self.sensor_info["frequency_bands"]:
                pow = self.get_sensor_band_power(pow_data, sensor, band)
                vals.append(pow)

            entropy = self.calc_spectral_entropy(vals)
            entropies.append(entropy)

        return float(np.mean(entropies))

    def calc_hjorth_mobility_proxy(self, pow_data: np.ndarray) -> float:
        """
        Frequency-domain Hjorth mobility approximation.

        Original time-domain Hjorth mobility:
            sqrt(var(dx/dt) / var(x))

        Power-domain approximation:
            sqrt(m2 / m0)

        where:
            m0 = sum(P)
            m2 = sum(f^2 * P)
        """
        band_powers = self.aggregate_power_by_band(pow_data)

        m0 = 0.0
        m2 = 0.0

        for band, power in band_powers.items():
            center = self.band_frec_centrers[band]
            m0 += power
            m2 += center**2 * power

        return float(np.sqrt(m2 / (m0 + self.eps)))

    def calc_hjorth_complexity_proxy(self, pow_data: np.ndarray) -> float:
        """
        Frequency-domain Hjorth complexity approximation.
            complexity ~= sqrt((m4 * m0) / m2^2)

        where:
            m0 = sum(P)
            m2 = sum(f^2 * P)
            m4 = sum(f^4 * P)
        """
        band_powers = self.aggregate_power_by_band(pow_data)

        m0 = 0.0
        m2 = 0.0
        m4 = 0.0

        for band, power in band_powers.items():
            center = self.band_frec_centrers[band]
            m0 += power
            m2 += center**2 * power
            m4 += center**4 * power

        if m0 <= self.eps or m2 <= self.eps:
            return np.nan

        return float(np.sqrt((m4 * m0) / (m2**2 + self.eps)))

    def calc_median_frequency_proxy(self, pow_data: np.ndarray) -> float:
        """
        Approximate median frequency from band centers.

        Returns the center frequency of the band where cumulative power reaches
        50 percent of total power.
        """
        band_powers = self.aggregate_power_by_band(pow_data)

        items = sorted(
            band_powers.items(),
            key=lambda item: self.band_frec_centrers[item[0]],
        )

        total = sum(power for _, power in items)

        cumulative = 0.0

        for band, power in items:
            cumulative += power

            if cumulative >= total / 2.0:
                return float(self.band_frec_centrers[band])

        # Return highest band center if something goes wrong
        return float(self.band_frec_centrers[items[-1][0]])

    # TODO: check if this makes sense
    def calc_diffuse_slowing_proxy(self, pow_data: np.ndarray) -> float:
        """
        Original diffuse slowing often depends on delta/theta activity.

        Emotiv vector has no delta, so this proxy uses:

            theta / (alpha + betaL + betaH)
        """
        theta = self.calc_sum_band_power(pow_data, "theta")
        alpha = self.calc_sum_band_power(pow_data, "alpha")
        betal = self.calc_sum_band_power(pow_data, "betaL")
        betah = self.calc_sum_band_power(pow_data, "betaH")

        denominator = alpha + betal + betah

        return float(theta / (denominator + self.eps))

    def calc_mean_band_power(self, pow_data: np.ndarray, band: str) -> float:
        values = self.get_band_values(pow_data, band)
        return float(np.mean(values))

    def calc_sum_band_power(self, pow_data: np.ndarray, band: str) -> float:
        values = self.get_band_values(pow_data, band)
        return float(np.sum(values))

    def calc_relative_band_power(self, pow_data: np.ndarray, band: str) -> float:
        band_power = self.calc_sum_band_power(pow_data, band)
        total_power = self.calc_total_power(pow_data)
        return float(band_power / (total_power + self.eps))

    def calc_ratio(
        self,
        pow_data: np.ndarray,
        numerator_band: List[str],
        denominator_band: List[str],
    ) -> float:
        numerator = 0.0
        for band in numerator_band:
            numerator += self.calc_sum_band_power(pow_data, band)

        denominator = 0.0
        for band in denominator_band:
            denominator += self.calc_sum_band_power(pow_data, band)

        return float(numerator / (denominator + self.eps))

    def calc_siq_proxy(self, pow_data: np.ndarray, band: str) -> float:
        """
        Sub-band information quantity proxy.

        Original SIQ is entropy of the filtered time-domain signal. With only
        band power, a reasonable proxy is the entropy contribution of the
        relative band power:

            -p_band * log(p_band)
        """
        p_band = self.calc_relative_band_power(pow_data, band)
        return float(-p_band * np.log(p_band + self.eps))

    def calc_mean_differential_entropy_proxy(
        self,
        pow_data: np.ndarray,
        band: str,
    ) -> float:
        """Mean differential entropy proxy for a band."""
        values = self.get_band_values(pow_data, band)
        de_values = [self.differential_entropy_proxy(val) for val in values]
        return float(np.mean(de_values))

    def differential_entropy_proxy(self, power: float) -> float:
        """
        Gaussian differential entropy approximation.

        If EEG band power approximates variance:

            h(x) = 0.5 * log(2 * pi * e * variance)
        """
        power = max(float(power), self.eps)
        return float(0.5 * np.log(2.0 * np.pi * np.e * power))

    def get_band_values(self, pow_data: np.ndarray, band: str) -> List[float]:
        """Get power values for all sensors for a specific band."""
        values = []
        for sensor in self.sensor_info["electrodes"]:
            idx = self.sensor_band_idx[(sensor, band)]
            values.append(float(pow_data[idx]))

        return values

    def get_sensor_band_power(
        self, pow_data: np.ndarray, sensor: str, band: str
    ) -> float:
        """Get power value for a specific sensor/band combination."""
        idx = self.sensor_band_idx[(sensor, band)]
        return float(pow_data[idx])

    def aggregate_power_by_band(self, pow_data: np.ndarray) -> Dict[str, float]:
        """Sum power by exact band (no grouping)."""
        band_powers = {}
        for band in self.sensor_info["frequency_bands"]:
            band_powers[band] = 0.0
            indices = self.band_indices[band]
            for idx in indices:
                band_powers[band] += float(pow_data[idx])

        return band_powers
