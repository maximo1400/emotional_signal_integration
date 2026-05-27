# Emotional Signal Integration

This repository provides a layered pipeline for extracting EEG band-power features (Layer 1), either simulate emotional-state data or capture from an EEG device. Preparing for downstream layers that will classify valence/arousal (Layer 2)(WIP) and broadcast results over sockets for integration with other software.

## Quick overview
- Purpose: capture or simulate band-power features, normalize valence/arousal, and stream L1 output for downstream processing.
- Layers:
	- L1 — Band-power capture & simulator (`L1_band_power_capture/`). Produces power vectors and an `output_df` when the simulator finishes.
	- L2 (future) — VA classifier: consumes L1 vectors, produces `valence`, `arousal`, `label`, `confidence` and publishes over sockets.

## Repo layout
- `config.yml` — main configuration (data source, sequences, emotion ranges).
- `main.py` — top-level orchestrator.
- `requirements.txt` — Python dependencies.
- `L1_band_power_capture/` — Layer 1 code (Emotiv connectors, simulator, data adapters):
	- `Emotiv/` — live capture helpers and connectors.
	- `Simulated_pow/` — `EmotionSimulator.py`, toy datasets in `Data/`.
	- `DB_adaptation/` — converters between formats (MAT → feather, EEG → pow, etc.).
- `output_data/` — output and exported artifacts.

## What L1 produces
- A stream (Python `queue.Queue`) of power vectors ordered by `POW_COLUMNS` (see `L1_band_power_capture/Simulated_pow/EmotionSimulator.py`).
- When the simulator finishes (or on demand), `EmotionSimulator.output_df` contains rows with columns in this order:

	power columns → `valence` → `arousal` → `emot_state` → `smoothed` → `timestamp`

- `valence` and `arousal` are normalized to `emotion_range` from `config.yml`. `smoothed` is `True` if the values were blended during a transition between states.

## How to run

1) Install dependencies:
```bash
pip install -r requirements.txt
```
2) Configure `config.yml` as needed (e.g., choose `pow_data_source`, set sequences, etc.).

3) If using the virtual simulator, or need to train a classifier, ensure you have the necessary data files in `Dreamer/Data`. 
- Get access to the [Dreamer](https://zenodo.org/records/546113) 
 dataset and place the relevant `.mat` files in that directory.
  ```bash
  uv run python Dreamer/DB_adaptation/run_adaptation.py
  ```
- Alternatively, you can use the legacy script with `torcheeg` by running:
  ```bash
  cd Dreamer/DB_adaptation/torcheeg_legacy
  uv run python run_torcheeg_adaptation.py
  ```

4) If using the Emotiv device, ensure you have the necessary hardware set up and  add yor app credentials in `.env` file, it should look like:
```.env
APP_CLIENT_ID = "you_client_id"
APP_CLIENT_SECRET = "your_client_secret"
```


5) Orchestrate via `main.py`:
```bash
python main.py
```
You can overwrite any setting from `config.yml` by passing it as a command line flag using the standard `--key value` syntax. For example:
```bash
python main.py --verbose True --classifier_mode train
```
Not recommended for bigger changes, but useful for quick overrides.

## Key `config.yml` settings (summary)
- `pow_data_source`: `"virtual"` or `"emotiv"`.
- `sub_id`: subject filter for virtual data; `-1` uses all subjects.
- `transition_duration`: seconds to blend between states (0 = instant).
- `sequences`: lists of `[state_id, duration_seconds]` pairs.
- `emotion_range`: e.g. `[-1.0, 1.0]` used to normalize VA values.
- `emotional_states_areas`: defines VA rectangles; rows outside these get state `NA`.

## Output schema (per timestep row)
- Power columns (floats) in order defined by `POW_COLUMNS`.
- `valence` (float) — normalized to `emotion_range`.
- `arousal` (float) — normalized to `emotion_range`.
- `emot_state` (string) — state id assigned from `emotional_states_areas`.
- `smoothed` (bool) — `True` if blended during a transition.
- `timestamp` (float) — UNIX epoch seconds.

## Layer 2 (classifier) and socket integration (design notes) (WIP)
- L2 consumes L1 power vectors and outputs VA estimates. Consider:
	- A lightweight scikit-learn model or a small neural net that uses a short temporal window.
	- Output payload: `{source, sub_id, timestamp, valence, arousal, label, confidence}`.

- `L2_emot_state_estimation/run.py` can now train a fresh classifier from labeled Feather data when no `--model-path` is provided, and it saves the fitted model next to the prediction outputs by default.
- To reuse an existing model, pass `--model-path /path/to/classifier.joblib`.

- Socket transport options being explored:
	- TCP with newline-delimited JSON (simple, cross-language).
	- WebSocket (if browser/HTTP clients required).

- Example minimal JSON message:
```json
{
	"source": "L2_classifier",
	"sub_id": 4,
	"timestamp": 1680000000.0,
	"valence": 0.12,
	"arousal": -0.45,
	"label": "neutral",
	"confidence": 0.78
}
```
