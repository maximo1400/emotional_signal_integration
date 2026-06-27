# DREAMER DB Adaptation

This module handles the extraction and transformation of the raw DREAMER dataset (MATLAB format) into a fast, pandas-friendly Feather format containing calculated EEG band power features.

## Purpose

The original DREAMER dataset stores raw EEG readings in a large `.mat` file. Our pipeline operates on band power features in specific frequency bands (Theta, Alpha, BetaL, BetaH, Gamma). This module bridges that gap by:
1. Parsing the `.mat` file into raw dataframes.
2. Processing the raw EEG signals into the relevant frequency band power features using spectral analysis.
3. Saving the results as a highly optimized `.feather` file for rapid loading by the virtual simulator (Layer 1) and classifier training (Layer 2).

## Structure

- `run_adaptation.py` — The main orchestrator script. Reads paths from `config.yml`, executes the parsing and transformation steps, and writes the output `.feather` file.
- `mat_to_df.py` — Contains the logic to extract raw EEG arrays and emotional state ratings from the complex MATLAB structure into structured Pandas DataFrames.
- `eeg_to_pow.py` — Contains the signal processing logic to convert the raw EEG timeseries into band power features.

## Usage

This adaptation only needs to be run once, assuming you have downloaded `DREAMER.mat` into the `Dreamer/` directory. running `main.py`  in a mode requiring the virtual dataset (`pow_data_source: virtual` or `classifier_mode: train`) and the configured `.feather` file is missing, it will automatically execute this adaptation step.

The file location is set in `config.yml` under `dreamer_feather_path`. It may be needed to delette the file if the file fails to load, as feather files are not guaranteed to be compatible across different versions of Pandas. If you need to re-run the adaptation, simply delete the existing `.feather` file and run `main.py` again.

## Signal Processing Pipeline (Mathematical Explanation)

The conversion from raw EEG voltage (µV) to frequency band power inside `eeg_to_pow.py` rigorously mimics the Emotiv EPOC+ signal processing chain. Here is the mathematical breakdown of each frame's processing:

### 1. Slew-Rate Clipping
To prevent high-frequency artifacts introduced by sudden extreme voltage spikes (e.g., from physical bumps), the sample-to-sample difference is calculated and clamped to a maximum $\delta$ (30 µV).
$$ \Delta x[n] = x[n] - x[n-1] $$
$$ \Delta x_{clipped}[n] = \max(-\delta, \min(\delta, \Delta x[n])) $$
The signal is then reconstructed via cumulative sum:
$$ x_{clipped}[n] = \sum_{i=0}^n \Delta x_{clipped}[i] $$

### 2. High-Pass Filter (0.5 Hz)
A 2nd-order Butterworth IIR high-pass filter removes DC drift and ultra-low frequency sweat artifacts.
$$ y[n] = \sum_{k=0}^2 b_k x_{clipped}[n-k] - \sum_{k=1}^2 a_k y[n-k] $$

### 3. Epoching
The continuous signal is divided into overlapping frames. 
- **Window size ($N$)**: 256 samples (2.0 seconds at 128 Hz)
- **Hop size**: 16 samples (0.125 seconds)

### 4. Robust DC Removal (Interquartile Mean)
To center the signal around zero without being skewed by large localized artifacts within an epoch, the Interquartile Mean (IQM) is subtracted instead of a standard arithmetic mean.
$$ Q_1, Q_3 = \text{25th and 75th percentiles of the epoch } X $$
$$ X_{IQR} = \{ x \in X \mid Q_1 \le x \le Q_3 \} $$
$$ \text{IQM} = \frac{1}{|X_{IQR}|} \sum_{x \in X_{IQR}} x $$
$$ X_{centered} = X - \text{IQM} $$

### 5. Hann Windowing
A raised cosine window is applied to minimize spectral leakage at the edges of the finite epoch. The Emotiv implementation uses a scaling factor of 2.0.
$$ w[n] = 0.5 \left( 1 - \cos\left(\frac{2\pi (n+1)}{N+1}\right) \right) $$
$$ X_{windowed}[n] = X_{centered}[n] \times 2.0 \times w[n] $$

### 6. Power Spectrum (FFT)
The Fast Fourier Transform (FFT) is used to transition to the frequency domain.
$$ X(k) = \sum_{n=0}^{N-1} X_{windowed}[n] e^{-i 2\pi k n / N} $$
The absolute power for each frequency bin $k$ is calculated by squaring the magnitude and dividing by the window length $N$:
$$ P(k) = \frac{|X(k)|^2}{N} $$

### 7. Band Aggregation and Logarithmic Scaling
The powers of the frequency bins that fall within each designated frequency band (e.g., Theta: 4–8 Hz) are summed together.
$$ P_{band} = \sum_{f_k \in [f_{low}, f_{high})} P(k) $$
To output in logarithmic scale (closer to human perception / standard feature representation), a base-10 logarithm is applied with a tiny epsilon to avoid $\log(0)$:
$$ P_{out} = \log_{10}(P_{band} + 10^{-20}) $$
