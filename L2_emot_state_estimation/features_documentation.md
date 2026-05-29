# FeatureSelector Documentation

The `FeatureSelector` class is responsible for expanding raw Emotiv band-power vectors into a rich set of derived features. This expanded feature space provides more robust and meaningful inputs for machine learning models attempting to estimate emotional states, specifically **Valence** and **Arousal**.

Since the input consists solely of frequency-domain band-power data (theta, alpha, betaL, betaH, gamma for 14 sensors), many traditional time-domain features (like true Hjorth parameters or exact signal entropy) cannot be computed directly. Instead, `FeatureSelector` computes mathematically sound **proxies** using the available spectral information.

---

## Supported Features and Their Meanings

The features that can be added via the `features_to_add` and `asymmetries` configurations are grouped into the following categories:

### 1. Global Power and Statistical Features
These features capture the overall magnitude and variance of the EEG signal across all sensors and bands.
* **`total_power`**: The sum of all power values across all bands and sensors.
* **`log_total_power`**: The natural logarithm of the total power, often more normally distributed.
* **`power_mean`**: The average power across the 70-dimensional vector.
* **`power_std`**: The standard deviation of the power vector.
* **`signal_std`**: A proxy for the standard deviation of the original time-domain signal, approximated as the square root of the total power.

### 2. Entropy Features
Entropy measures the unpredictability, randomness, or complexity of the signal. Higher entropy usually signifies higher cognitive processing or a more active brain state.
* **`shannon_entropy` / `spectral_entropy`**: The Shannon entropy calculated over the normalized 70-dimensional power vector, scaled between 0 and 1.
* **`mean_sensor_spectral_entropy`**: Calculates the spectral entropy for *each* sensor individually across its 5 frequency bands, and then averages them. This often provides a more localized measure of complexity than global entropy.

### 3. Hjorth Parameter Proxies
Hjorth parameters are traditionally time-domain descriptors of an EEG trace. Here, they are approximated in the frequency domain using spectral moments ($m_0, m_2, m_4$).
* **`hjorth_mobility`**: Approximates the mean frequency of the signal. It represents the standard deviation of the slope with reference to the standard deviation of the amplitude.
* **`hjorth_complexity`**: Approximates the change in frequency. It measures how the shape of the signal deviates from a pure sine wave.

### 4. Frequency & Slowing Proxies
* **`median_frequency`**: The frequency below which 50% of the total signal power is contained, approximated using the center frequencies of the Emotiv bands.
* **`diffuse_slowing`**: A proxy for general brain slowing, calculated as `theta / (alpha + beta)`. High diffuse slowing can indicate fatigue, drowsiness, or certain pathological states.

### 5. Band Ratios and Engagement
Ratios between different frequency bands are some of the most established features in BCI for measuring cognitive and emotional states.
* **`engagement_index`**: Calculated as `beta / (alpha + theta)`. Widely used to measure alertness, attention, and cognitive workload.
* **`beta_alpha_ratio`**: `beta / alpha`. A classic measure of arousal and cortical activation.
* **`theta_beta_ratio`**: `theta / beta`. Often used in attention deficit studies; inversely related to arousal.
* **`theta_alpha_ratio`**: `theta / alpha`. Can reflect drowsiness or relaxation.
* **`gamma_beta_ratio`**: `gamma / beta`.
* **`slow_fast_ratio`**: The ratio of slow waves (`theta`) to fast waves (`alpha`, `betaL`, `betaH`, `gamma`).

### 6. Hemispheric Asymmetries
Calculated between pairs of electrodes on the left and right hemispheres (e.g., F3/F4, AF3/AF4). The `FeatureSelector` supports multiple methods for calculating this difference:
* **`difference`**: `mean(Left - Right)`
* **`ratio`**: `mean(Left / Right)`
* **`log_ratio`**: `mean(log(Left) - log(Right))`
* **`dasm`** (Differential Asymmetry): The difference in Differential Entropy (DE) between left and right hemispheres.
* **`rasm`** (Rational Asymmetry): The ratio of DE between left and right hemispheres.

### 7. Band-Specific Proxies
These features can be extracted for any specific band (`theta`, `alpha`, `betaL`, `betaH`, `beta`, `gamma`):
* **`mean_<band>`**: Mean power across all sensors for the given band.
* **`relative_<band>`**: The band's power divided by the total power (e.g., relative alpha).
* **`siq_<band>`**: Sub-band Information Quantity proxy (`-p * log(p)`).
* **`de_<band>_mean`**: Mean Differential Entropy proxy for the specific band. DE is approximated as a Gaussian distribution based on variance (power).

---

## Utility in Estimating Valence and Arousal

Transitioning from raw band power data to emotional dimensions (Valence and Arousal) is notoriously difficult. The features added by `FeatureSelector` are highly beneficial for this translation for the following reasons:

### Predicting Valence (Positivity vs. Negativity)
Valence is most strongly correlated with **frontal EEG asymmetry**. The left hemisphere is generally associated with "approach" behaviors and positive emotions, while the right hemisphere is associated with "withdrawal" behaviors and negative emotions.
* **How `FeatureSelector` helps**: The **Asymmetry features** (`dasm`, `rasm`, `difference`, `log_ratio`) specifically target this phenomenon. By configuring frontal pairs (e.g., F3-F4, AF3-AF4), the model is fed a direct mathematical representation of hemispheric dominance, usually in the Alpha or Beta bands, making Valence estimation much more feasible.

### Predicting Arousal (Excitement vs. Calmness)
Arousal is linked to general cortical activation, alertness, and cognitive load. High arousal states typically exhibit higher power in fast frequency bands (Beta, Gamma) and lower power in slow bands (Theta, Alpha).
* **How `FeatureSelector` helps**: 
    1. **Band Ratios**: Features like `beta_alpha_ratio` and the `engagement_index` are classic indicators of arousal. They condense the relationship between activating (beta) and relaxing (alpha/theta) bands into a single, highly correlated metric.
    2. **Complexity and Entropy**: High arousal often leads to a more complex, less predictable EEG signal. Features like `spectral_entropy`, `hjorth_mobility`, and `hjorth_complexity` capture this desynchronization.
    3. **Global Activity**: Metrics like `total_power` and `avg_frontal_beta` can provide a baseline for overall brain activity levels during an emotional response.

**Summary**: By using `FeatureSelector`, the machine learning pipeline is not forced to implicitly learn complex neurophysiological relationships (like "beta divided by alpha") from raw arrays. Instead, it is explicitly provided with these established psychophysiological biomarkers, drastically improving the model's ability to map EEG data to the Valence-Arousal space.
