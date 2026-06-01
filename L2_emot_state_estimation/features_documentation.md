# FeatureSelector Documentation

The `FeatureSelector` class is responsible for expanding raw Emotiv band-power vectors into a rich set of derived features. This expanded feature space provides more robust and meaningful inputs for machine learning models attempting to estimate emotional states, specifically **Valence** and **Arousal**.

Since the input consists solely of frequency-domain band-power data (theta, alpha, betaL, betaH, gamma for 14 sensors), many traditional time-domain features (like exact signal entropy) cannot be computed directly. Instead, `FeatureSelector` computes mathematically sound **proxies** using the available spectral information.

---

## Supported Features and Their Meanings

The features that can be added via the `features_to_add` and `asymmetries` configurations are grouped into the following categories:

### 1. Global Power and Statistical Features
These features capture the overall magnitude and variance of the EEG signal across all sensors and bands.
* **`total_power`**: The sum of all power values across all bands and sensors.
* **`power_mean`**: The average power across the 70-dimensional vector.
* **`power_std`**: The standard deviation of the power vector.

### 2. Entropy Features
Entropy measures the unpredictability, randomness, or complexity of the signal. Higher entropy usually signifies higher cognitive processing or a more active brain state.
* **`spectral_entropy`**: The Shannon entropy calculated over the normalized 70-dimensional power vector, scaled between 0 and 1.
* **`mean_sensor_spectral_entropy`**: Calculates the spectral entropy for *each* sensor individually across its 5 frequency bands, and then averages them. This often provides a more localized measure of complexity than global entropy.

### 3. Frequency Proxies
* **`spectral_centroid`**: The center of mass of the spectrum, calculated as the weighted mean of the frequencies present in the signal, where their weights are their power.

### 4. Hemispheric Asymmetries
Calculated between pairs of electrodes on the left and right hemispheres (e.g., F3/F4, AF3/AF4). The `FeatureSelector` supports multiple methods for calculating this difference:
* **`difference`**: `mean(Left - Right)`
* **`ratio`**: `mean(Left / Right)`
* **`log_ratio`**: `mean(log(Left) - log(Right))`
* **`dasm`** (Differential Asymmetry): The difference in Differential Entropy (DE) between left and right hemispheres.
* **`rasm`** (Rational Asymmetry): The ratio of DE between left and right hemispheres.

### 5. Band-Specific Proxies
These features can be extracted for any specific band (`theta`, `alpha`, `betaL`, `betaH`, `beta`, `gamma`):
* **`mean_<band>`**: Mean power across all sensors for the given band.
* **`relative_<band>`**: The band's power divided by the total power (e.g., relative alpha).
* **`band_entropy_contribution_<band>`**: Entropy contribution of the relative band power (`-p * log(p)`). It is a very loose proxy for SIQ (Sub-band Information Quantity).
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
    1. **Complexity and Entropy**: High arousal often leads to a more complex, less predictable EEG signal. Features like `spectral_entropy` capture this desynchronization.
    2. **Global Activity**: Metrics like `total_power` and `avg_frontal_beta` can provide a baseline for overall brain activity levels during an emotional response.

**Summary**: By using `FeatureSelector`, the machine learning pipeline is not forced to implicitly learn complex neurophysiological relationships (like "beta divided by alpha") from raw arrays. Instead, it is explicitly provided with these established psychophysiological biomarkers, drastically improving the model's ability to map EEG data to the Valence-Arousal space.
