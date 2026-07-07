# Emotional Signal Integration

This repository provides a layered, end-to-end pipeline for capturing EEG band-power features, estimating emotional states (Valence/Arousal), and broadcasting the results over sockets for downstream integration (e.g., Unity, other applications). 

## How to run

1) Install dependencies using `uv` (recommended) or `pip`:
```bash
uv sync
```

2) Configure `config.yml` as needed (choose `pow_data_source`, classifier, socket config, etc.).

3) *Optional (For virtual simulator or training classifiers)*: Ensure you have the DREAMER dataset.
- Place the relevant `.mat` files in `Dreamer/`.

4) *Optional (For live Emotiv headset)*: Add your app credentials in `.env` file:
```.env
APP_CLIENT_ID="your_client_id"
APP_CLIENT_SECRET="your_client_secret"
```

5) Run the full pipeline via `main.py`:
```bash
uv run main.py
```
You can overwrite most settings from `config.yml` by passing it as a command line flag using the standard `--key value` syntax. For example:
```bash
uv run python main.py --verbose True --classifier_mode train
```

## Key Configuration (`config.yml`)
- `pow_data_source`: `"virtual"` or `"emotiv"`. (chooses between simulated DREAMER data or live Emotiv headset capture)
- `classifier_mode`: `"train"`, `"predict_from_queue"`, or `"predict_from_file"`.
- `classifier`: Model to use (e.g., `"knn"`, `"random_forest"`, `"svm"`, `"balanced_random_forest"`).
- [`features_to_add` & `asymmetries`](L2_emot_state_estimation/features_documentation.md): Extracted EEG features used in classification.
- `smoothing_method`: L3 smoothing strategy (`"rolling_buffer"`, `"ema"`, `"steps"`, `"none"`).


## Pipeline Architecture
The pipeline is divided into three processing layers:

### [Layer 1: Band Power Capture](L1_band_power_capture/README.md) (`L1_band_power_capture/`)
Captures band-power features from an Emotiv EEG device or simulates data based on the DREAMER dataset. 
- Produces power vectors and raw data frames.
- Features include: Live streaming from Emotiv (`pow_data_source: emotiv`), Virtual sequence simulation (`pow_data_source: virtual`), and data adaptation.

### [Layer 2: Emotional State Estimation](L2_emot_state_estimation/README.md) (`L2_emot_state_estimation/`)
Consumes L1 power vectors and outputs Valence/Arousal (VA) predictions, categorical emotion labels, and confidence metrics.
- Uses machine learning classifiers (KNN, Random Forest, SVM) to predict states.
- Supports extracting advanced features (e.g., Frontal Alpha Asymmetry, Spectral Entropy).
- Capable of live predictions from queue or batch predictions from file.
- Includes a training module to train a fresh classifier on the DREAMER dataset.

### [Layer 3: VA Data Adaptation & Output](L3_va_data_adaptation/README.md) (`L3_va_data_adaptation/`)
Consumes L2 predictions, smooths the signals, and broadcasts the data.
- **Smoothing Algorithms**: Moving average (rolling buffer), Exponential Moving Average (EMA), Step-based jumps.
- **Socket Integration**: Broadcasts predictions over TCP or UDP for real-time external integration.
- Outputs can also be saved to the `output_data/` directory.

**_By default L3 expecets a listener to connect to it using TCP. You can change the host, port, and protocol in `config.yml` as needed or deactivate the listener if external one is used._**

## Repository Layout
- `config.yml` — Main configuration (data source, sequences, ML models, sockets).
- `main.py` — Top-level orchestrator that connects the queues of all three layers.
- `requirements.txt` / `pyproject.toml` — Python dependencies (managed via `uv`).
- `config_loader.py` — Centralized configuration parsing.
- `output_listener.py` — Standalone socket client for testing Layer 3 broadcasts.
- `output_data/` — Directory for all exported artifacts and CSV outputs across layers, if `save_output_files` value is set in `config.yml`.
- [`Dreamer/DB_adaptation/`](Dreamer/DB_adaptation/README.md) — Code to adapt the DREAMER dataset for virtual simulation and L2 model training.

## Output Schema
The L3 socket broadcasts a JSON payload for every timestep:
```json
{
  "valence": 0.11,
  "arousal": -0.43,
  "confidence": 0.78,
  "timestamp": 1680000000.0
}
```
