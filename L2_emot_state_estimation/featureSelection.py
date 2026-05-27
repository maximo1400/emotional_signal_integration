import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np

# Add parent directory to path to import config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config


class FeatureSelector:
    """
    Feature selector / feature expander for Emotiv band-power vectors.

    Input pow_data must be length 70 and ordered according to POW_COLUMNS:

        sensor/theta, sensor/alpha, sensor/betaL, sensor/betaH, sensor/gamma

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

    Features that cannot be computed from power-only data:

    - False nearest neighbor
    - Spikes
    - Sharp spikes
    - Delta burst after spike
    - Number of bursts
    - Burst length mean/std
    - Number of suppressions
    - Suppression length mean/std
    """

    BAND_GROUPS: Dict[str, List[str]] = {
        "theta": ["theta"],
        "alpha": ["alpha"],
        "beta": ["betaL", "betaH"],
        "betaL": ["betaL"],
        "betaH": ["betaH"],
        "gamma": ["gamma"],
        "slow": ["theta"],
        "fast": ["alpha", "betaL", "betaH", "gamma"],
    }

    # Approximate center frequencies for Emotiv bands.
    # Used for Hjorth and median-frequency proxies.
    BAND_CENTERS: Dict[str, float] = {
        "theta": 6.0,
        "alpha": 10.0,
        "betaL": 16.0,
        "betaH": 25.0,
        "gamma": 37.5,
    }

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
        self.eps = 1e-10
        self.pow_columns_mask_np = np.asarray([], dtype=bool)

        self.load_config(features)
        self.labels = self.get_final_feature_names()

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

        self.pow_columns_mask_np = np.asarray(self.pow_columns_mask, dtype=bool)

        self.band_indices = {}
        for band in self.sensor_info["frequency_bands"]:
            self.band_indices[band] = np.asarray(
                [
                    self.col_index[f"{sensor}/{band}"]
                    for sensor in self.sensor_info["electrodes"]
                ],
                dtype=int,
            )

        self.sensor_band_indices = {}
        for sensor in self.sensor_info["electrodes"]:
            for band in self.sensor_info["frequency_bands"]:
                col = f"{sensor}/{band}"
                idx = self.col_index.get(col)
                self.sensor_band_indices[(sensor, band)] = idx

    def process_data(self, pow_data) -> List[float]:
        if len(pow_data) != len(self.pow_columns):
            raise ValueError(
                f"Expected pow_data length {len(self.pow_columns)}, "
                f"got {len(pow_data)}."
            )

        pow_arr = np.asarray(pow_data, dtype=float)

        data = pow_arr[self.pow_columns_mask_np].tolist()
        data.extend(self.add_features(pow_arr))

        return data

    def get_final_feature_names(self) -> List[str]:
        feature_names = [
            col for col, keep in zip(self.pow_columns, self.pow_columns_mask) if keep
        ]

        for area, asym_type, band in self.asymmetries:
            feature_names.append(f"{area}_{asym_type}_{band}_asymmetry")

        for feat in self.features_to_add:
            feature_names.append(feat)

        feature_names.append("timestamp")
        return feature_names

    def add_features(self, pow_data) -> List[float]:
        features = []

        for area, asym_type, band in self.asymmetries:
            features.append(self.calc_asymmetry(pow_data, area, asym_type, band))

        for feat in self.features_to_add:
            features.append(self.calc_feature(pow_data, feat))

        return features

    def calc_feature(self, pow_data, feat: str) -> float:
        """
        Supported feature names:

        Global power/statistical:
            total_power
            log_total_power
            power_mean
            power_std
            signal_std

        Entropy:
            shannon_entropy
            spectral_entropy
            mean_sensor_spectral_entropy

        Hjorth proxies:
            hjorth_mobility
            hjorth_complexity

        Frequency proxy:
            median_frequency

        Slowing/ratios:
            diffuse_slowing
            engagement_index
            beta_alpha_ratio
            theta_beta_ratio
            theta_alpha_ratio
            gamma_beta_ratio
            slow_fast_ratio

        Band means:
            mean_theta
            mean_alpha
            mean_beta
            mean_betaL
            mean_betaH
            mean_gamma

        Relative powers:
            relative_theta
            relative_alpha
            relative_beta
            relative_betaL
            relative_betaH
            relative_gamma

        SIQ proxies:
            siq_theta
            siq_alpha
            siq_beta
            siq_betaL
            siq_betaH
            siq_gamma

        Differential entropy proxies:
            de_theta_mean
            de_alpha_mean
            de_beta_mean
            de_betaL_mean
            de_betaH_mean
            de_gamma_mean

        Custom:
            avg_frontal_beta
        """
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
            return self.calc_ratio(pow_data, "beta", ["alpha", "theta"])

        if lname == "beta_alpha_ratio":
            return self.calc_ratio(pow_data, "beta", "alpha")

        if lname == "theta_beta_ratio":
            return self.calc_ratio(pow_data, "theta", "beta")

        if lname == "theta_alpha_ratio":
            return self.calc_ratio(pow_data, "theta", "alpha")

        if lname == "gamma_beta_ratio":
            return self.calc_ratio(pow_data, "gamma", "beta")

        if lname == "slow_fast_ratio":
            return self.calc_ratio(pow_data, "slow", "fast")

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
        pow_data,
        area: str,
        diff_method: str,
        frec_band: str,
        eps: float = 1e-10,
    ) -> float:
        """
        Calculates asymmetry over configured left/right electrode pairs.

        With your config:

            frontal_pairs:
                [F3, F4]
                [AF3, AF4]
                [F7, F8]

            parietal_pairs:
                [P7, P8]

        Supported diff_method values:

            difference:
                mean(left - right)

            ratio:
                mean(left / right)

            log_ratio:
                mean(log(left) - log(right))

            dasm:
                Differential asymmetry approximation:
                mean(DE(left) - DE(right))

            rasm:
                Rational asymmetry approximation:
                mean(DE(left) / DE(right))

        DE is approximated from power as:

            DE ~= 0.5 * log(2 * pi * e * power)
        """
        pairs = self.get_area_pairs(area)

        if len(pairs) == 0:
            return np.nan

        values = []

        for left, right in pairs:
            left_power = self.get_sensor_band_power(pow_data, left, frec_band)
            right_power = self.get_sensor_band_power(pow_data, right, frec_band)

            left_power = max(float(left_power), eps)
            right_power = max(float(right_power), eps)

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
                values.append(left_de / (right_de + eps))

            else:
                raise ValueError(f"Unknown asymmetry method: {diff_method}")

        if len(values) == 0:
            return np.nan

        return float(np.mean(values))

    def calc_avg_frontal_beta(self, pow_data) -> float:
        beta_values = []

        for electrode in self.sensor_info["frontal_electrodes"]:
            for band in ["betaL", "betaH"]:
                col_name = f"{electrode}/{band}"
                idx = self.col_index.get(col_name)
                beta_values.append(pow_data[idx])

        return float(np.mean(beta_values))

    def calc_total_power(self, pow_data) -> float:
        arr = np.maximum(pow_data, self.eps)
        return float(np.sum(arr))

    def calc_signal_std_proxy(self, pow_data) -> float:
        """
        Approximation.

        If summed band power approximates signal variance:

            std ~= sqrt(total_power)
        """
        return float(np.sqrt(self.calc_total_power(pow_data) + self.eps))

    def calc_spectral_entropy(self, pow_data) -> float:
        """
        Shannon entropy over the full 70-dimensional power vector.

        Normalized to [0, 1].
        """
        powers = np.maximum(pow_data, self.eps)
        probs = powers / (np.sum(powers) + self.eps)

        entropy = -np.sum(probs * np.log(probs + self.eps))
        max_entropy = np.log(len(probs))

        return float(entropy / (max_entropy + self.eps))

    def calc_mean_sensor_spectral_entropy(self, pow_data) -> float:
        """
        For each sensor, calculate entropy over its 5 bands, then average.

        This is often more meaningful than entropy over all 70 values.
        """
        entropies = []

        for sensor in self.sensor_info["electrodes"]:
            vals = []

            for band in self.sensor_info["frequency_bands"]:
                idx = self.sensor_band_indices.get((sensor, band))
                vals.append(pow_data[idx])

            vals = np.maximum(vals, self.eps)
            probs = vals / (np.sum(vals) + self.eps)

            entropy = -np.sum(probs * np.log(probs + self.eps))
            max_entropy = np.log(len(probs))

            entropies.append(entropy / (max_entropy + self.eps))

        return float(np.mean(entropies))

    def calc_hjorth_mobility_proxy(self, pow_data) -> float:
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
        band_powers = self.aggregate_power_by_exact_band(pow_data)

        m0 = 0.0
        m2 = 0.0

        for band, power in band_powers.items():
            center = self.BAND_CENTERS[band]
            m0 += power
            m2 += center**2 * power

        return float(np.sqrt(m2 / (m0 + self.eps)))

    def calc_hjorth_complexity_proxy(self, pow_data) -> float:
        """
        Frequency-domain Hjorth complexity approximation.

            complexity ~= sqrt((m4 * m0) / m2^2)

        where:

            m0 = sum(P)
            m2 = sum(f^2 * P)
            m4 = sum(f^4 * P)
        """
        band_powers = self.aggregate_power_by_exact_band(pow_data)

        m0 = 0.0
        m2 = 0.0
        m4 = 0.0

        for band, power in band_powers.items():
            center = self.BAND_CENTERS[band]
            m0 += power
            m2 += center**2 * power
            m4 += center**4 * power

        if m0 <= self.eps or m2 <= self.eps:
            return np.nan

        return float(np.sqrt((m4 * m0) / (m2**2 + self.eps)))

    # TODO: check return
    def calc_median_frequency_proxy(self, pow_data) -> float:
        """
        Approximate median frequency from band centers.

        Returns the center frequency of the band where cumulative power reaches
        50 percent of total power.
        """
        band_powers = self.aggregate_power_by_exact_band(pow_data)

        items = sorted(
            band_powers.items(),
            key=lambda item: self.BAND_CENTERS[item[0]],
        )

        total = sum(power for _, power in items)

        if total <= self.eps:
            return np.nan

        cumulative = 0.0

        for band, power in items:
            cumulative += power

            if cumulative >= total / 2.0:
                return float(self.BAND_CENTERS[band])

        return float(self.BAND_CENTERS[items[-1][0]])

    # TODO: check if this makes sense
    def calc_diffuse_slowing_proxy(self, pow_data) -> float:
        """
        Original diffuse slowing often depends on delta/theta activity.

        Emotiv vector has no delta, so this proxy uses:

            theta / (alpha + betaL + betaH)
        """
        theta = self.calc_sum_band_power(pow_data, "theta")
        alpha = self.calc_sum_band_power(pow_data, "alpha")
        beta = self.calc_sum_band_power(pow_data, "beta")

        denominator = alpha + beta

        return float(theta / (denominator + self.eps))

    def calc_mean_band_power(self, pow_data, band: str) -> float:
        values = self.get_band_values(pow_data, band)
        return float(np.mean(values))

    def calc_sum_band_power(self, pow_data, band: str) -> float:
        values = self.get_band_values(pow_data, band)
        return float(np.sum(values))

    def calc_relative_band_power(self, pow_data, band: str) -> float:
        band_power = self.calc_sum_band_power(pow_data, band)
        total_power = self.calc_total_power(pow_data)

        return float(band_power / (total_power + self.eps))

    def calc_ratio(
        self,
        pow_data,
        numerator_band: str,
        denominator_band: Union[str, Sequence[str]],
    ) -> float:
        numerator = self.calc_sum_band_power(pow_data, numerator_band)

        if isinstance(denominator_band, str):
            denominator = self.calc_sum_band_power(pow_data, denominator_band)
        else:
            denominator = 0.0

            for band in denominator_band:
                denominator += self.calc_sum_band_power(pow_data, band)

        return float(numerator / (denominator + self.eps))

    def calc_siq_proxy(self, pow_data, band: str) -> float:
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
        pow_data,
        band: str,
    ) -> float:
        values = self.get_band_values(pow_data, band)
        de_values = [self.differential_entropy_proxy(v) for v in values]

        return float(np.mean(de_values))

    def differential_entropy_proxy(self, power: float) -> float:
        """
        Gaussian differential entropy approximation.

        If EEG band power approximates variance:

            h(x) = 0.5 * log(2 * pi * e * variance)
        """
        power = max(float(power), self.eps)
        return float(0.5 * np.log(2.0 * np.pi * np.e * power))

    def get_band_values(self, pow_data, band: str) -> List[float]:
        bands = self.get_band_group(band)
        values = []

        for sensor in self.sensor_info["electrodes"]:
            for b in bands:
                idx = self.sensor_band_indices.get((sensor, b))
                values.append(float(pow_data[idx]))

        return values

    def get_sensor_band_power(self, pow_data, sensor: str, band: str) -> float:
        bands = self.get_band_group(band)
        total = 0.0

        for b in bands:
            idx = self.sensor_band_indices.get((sensor, b))
            total += float(pow_data[idx])

        return total

    def get_band_group(self, band: str) -> List[str]:
        return self.BAND_GROUPS.get(band, [band])

    # TODO: This probably should be done once on new pow_data, not separately for each feature that needs it.
    def aggregate_power_by_exact_band(self, pow_data) -> Dict[str, float]:
        band_powers = {band: 0.0 for band in self.sensor_info["frequency_bands"]}

        for sensor in self.sensor_info["electrodes"]:
            for band in self.sensor_info["frequency_bands"]:
                idx = self.sensor_band_indices.get((sensor, band))
                band_powers[band] += max(float(pow_data[idx]), self.eps)

        return band_powers

    def get_area_pairs(self, area: str) -> List[Tuple[str, str]]:
        if area == "frontal":
            return self.sensor_info.get("frontal_pairs", [])

        if area == "parietal":
            return self.sensor_info.get("parietal_pairs", [])

        if area == "all":
            return self.sensor_info.get("frontal_pairs", []) + self.sensor_info.get(
                "parietal_pairs", []
            )

        return []
