# Layer 1: Band Power Capture

L1 is responsible for sourcing the raw EEG band-power features that the rest of the pipeline uses to predict emotional states. It supports two modes of operation, defined by the `pow_data_source` setting in `config.yml`.

## Modes of Operation

1. **Virtual Simulation (`pow_data_source: virtual`)**
   - Driven by `EmotionSimulator.py`.
   - Simulates band-power features by sampling from the pre-processed DREAMER dataset.
   - You can configure the sequence of emotions to simulate and the duration of each in `config.yml`.
   - Emits a stream of power vectors that matches the expected Emotiv frequency format.

2. **Emotiv Live Capture (`pow_data_source: emotiv`)**
   - Utilizes the code in the `Emotiv/` directory.
   - Connects to an actual Emotiv headset via the Cortex API to stream live band-power (and other configured) data.
   - Requires valid `APP_CLIENT_ID` and `APP_CLIENT_SECRET` in a `.env` file.

## Structure

- `run.py` — The primary entry point for L1. Depending on the config, it initializes either the virtual simulator or the Emotiv live capture client and starts pumping data into the provided L1 output queue.
- `EmotionSimulator.py` — The virtual simulator logic. Handles loading the DREAMER feather file and smoothly transitioning between emotional states based on sequences.
- `Emotiv/` — Contains all the necessary adapters and Cortex API integrations for live headset capture.

## Output

L1 outputs a continuous stream of power vectors. Each vector contains 70 floating-point numbers ordered according to `POW_COLUMNS` defined in `config.yml`. This stream is passed directly to Layer 2 for feature extraction and classification. L1 also supports saving the captured or simulated data directly to the `output_data/` directory.
