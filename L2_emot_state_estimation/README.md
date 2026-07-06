# Layer 2: Emotional State Estimation

L2 consumes L1 power vectors and outputs Valence/Arousal predictions, categorical emotion labels, and confidence metrics based on trained machine learning models.

## Structure

- `classifier.py` — Contains the logic to load trained models, extract features from raw power vectors, and predict VA states.
- `featureSelection.py` — Module for extracting advanced EEG features, such as asymmetries and spectral entropy, from the L1 power vectors based on the configuration.
- `run.py` — Main execution script for L2. Handles training new models and running predictions (either from a live queue or a pre-recorded CSV).
- `EpochNormalizer.py` / `utils.py` — Utility scripts for data normalization, processing, and handling L2 input/output.
- `Models/` — Directory where trained classifier models (e.g., `knn.joblib`, `random_forest.joblib`) are saved and loaded from.
- `features_documentation.md` — Detailed explanation of the various EEG features that can be extracted.

## How It Works

L2 relies heavily on the `config.yml` at the root of the project to determine its behavior:
- **Normalization**: When processing data from an Emotiv EPOC headset and `enable_normalizer` is `True` in `config.yml`, the `EPOCCrossSessionNormalizer` is applied to incoming power vectors before feature extraction. It computes a global Z-score using `global_mu` and `global_sigma` from the config. If `simple_calibration` is `False`, it calibrates a session-specific baseline during the first `calibration_time` seconds of a session to remove cross-session offsets. If `simple_calibration` is `True`, it skips session calibration and uses the global baseline directly.
- **Feature Extraction**: Extracts specific subsets of the `POW_COLUMNS` and computes additional features as defined by `features_to_add` and `asymmetries`.
- **Classification**: Uses standard classifiers (KNN, Random Forest, SVM) trained to map these features to the Valence-Arousal space.
- **Modes**:
  - `train`: Uses the adapted DREAMER data to train models and save them to `Models/`.
  - `predict_from_queue`: Live processing mode; pulls power vectors from the L1 queue, makes predictions, and pushes them to the L3 queue.
  - `predict_from_file`: Batch processing mode; reads a pre-recorded L1 output CSV, makes predictions, and either saves or pushes them to L3.

## Usage

L2 is typically invoked automatically via the root `main.py` script. 

### Training a New Classifier

Make sure you have processed the DREAMER dataset (via Layer 1 adaptation), then run:
```bash
uv run main.py --mode train
```
Or simply set `classifier_mode: train` in `config.yml` and run `main.py`.

### Batch Processing (CSV Mode)

Process a saved L1 output CSV:

```bash
uv run main.py --mode predict_from_file --predict_from_file_pow_csv_path path/to/L1_output.csv
```
Again, you can also set `classifier_mode: predict_from_file` in `config.yml` and specify the CSV path.

### Real-Time Queue Integration

When running in `predict_from_queue` mode via `main.py`, L2 reads from the `l1_out` queue and puts a dictionary onto the `l2_out` queue formatted as:

```json
{
  "valence": 0.123,
  "arousal": -0.456,
  "label": "1_3",
  "confidence": 0.85,
  "timestamp": 1680000000.0
}
```
This payload is then consumed by L3 for smoothing and socket broadcasting.
