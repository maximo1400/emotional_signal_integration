# Layer 2: Emotional State Estimation

L2 takes L1 power vectors and outputs valence/arousal predictions + emotional state labels.

## Structure

- `classifier.py` — `VAClassifier` stub that predicts VA from power vectors
- `consumer.py` — `QueueConsumer` (real-time) and `CSVConsumer` (batch) to read L1 output
- `run.py` — Example runner for both queue and CSV modes

## Classifier (stub mode)

The `VAClassifier` is a skeleton / reference implementation:
- Input: power vector (70 floats, order = `POW_COLUMNS`)
- Output: `{valence, arousal, label, confidence}`

**Current implementation uses heuristics:**
- Arousal: alpha / (beta + theta) → normalized to [-1, 1]
- Valence: right-frontal minus left-frontal alpha → normalized to [-1, 1]
- Label: closest emotion in VA grid
- Confidence: stub value (0.65)

**Replace with a trained model:**
Replace the `predict()` method in `VAClassifier` with your own model (scikit-learn, PyTorch, etc.).

## Usage

### CSV mode (batch processing)

Process a saved L1 output CSV:

```bash
# Run L1 simulator, save output_df to CSV
python L1_band_power_capture/Simulated_pow/EmotionSimulator.py
# (quit after selecting sequence, output_df is in memory)

# Then process with L2 (requires output_df saved as CSV)
python L2_emot_state_estimation/run.py --mode csv --csv L1_output.csv
```

Output: `L2_predictions.csv` with columns:
- `index` — row number
- `true_emot_state` — ground truth from L1
- `true_valence`, `true_arousal` — ground truth VA from L1
- `pred_valence`, `pred_arousal` — L2 predictions
- `pred_label` — predicted emotion label
- `confidence` — prediction confidence
- `smoothed` — whether L1 smoothed values
- `timestamp` — sample timestamp

### Queue mode (real-time)

Connect L2 directly to L1 output queue (requires orchestration in `main.py`):

```python
from L1_band_power_capture.Simulated_pow.EmotionSimulator import EmotionSimulator
from L2_emot_state_estimation.classifier import VAClassifier
from L2_emot_state_estimation.consumer import QueueConsumer

import queue

# Run L1 simulator
q = queue.Queue()
sim = EmotionSimulator(q)
sim.main_loop()  # Runs in background or main thread

# Run L2 classifier on queue output
classifier = VAClassifier()
consumer = QueueConsumer(q)
for pow_vec in consumer.stream():
    result = classifier.predict(pow_vec)
    print(result)  # valence, arousal, label, confidence
```

## Example output (single prediction)

```json
{
  "valence": 0.123,
  "arousal": -0.456,
  "label": "neutral",
  "confidence": 0.65
}
```

## Improvements (TODOs)

- [ ] Train a real classifier on DREAMER or similar dataset
- [ ] Add support for temporal windowing (use last N frames, not just current)
- [ ] Add socket publisher (TCP JSON or ZeroMQ) to broadcast predictions
- [ ] Add metrics (accuracy vs true L1 labels, confusion matrix)
- [ ] Add model versioning and checkpointing
- [ ] Integrate with `main.py` for seamless L1 → L2 → publisher pipeline
