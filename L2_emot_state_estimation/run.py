"""
Layer 2 Runner: Example of how to use the classifier with L1 output

Modes:
1. queue_mode: Connect to live L1 output queue (real-time)
2. csv_mode: Process L1 output CSV (batch)
"""

import json
import os
import time
from pathlib import Path
import pyarrow.feather as feather
import yaml
import pandas as pd

from classifier import VAClassifier
from consumer import QueueConsumer, CSVConsumer

YAML_PATH = "config.yml"
OUTPUT_DIR = "L2_emot_state_estimation/output_data"


def load_yml_config(path: str | Path = YAML_PATH) -> dict:
    """Parse a YAML file and return the Python object it represents."""
    path = Path(path).expanduser()
    dict = {}
    yml_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    dict["feather_path"] = yml_data["feather_file_path"]
    dict["POW_COLUMNS"] = yml_data["POW_COLUMNS"]

    return dict


def run_feather_mode(feather_path: str, pow_columns: list):
    """
    Process L1 output from Feather file (DREAMER).

    Args:
        feather_path: Path to L1 output Feather file
    """
    print(f"Running L2 in Feather mode: {feather_path}")

    # Initialize classifier and consumer
    classifier = VAClassifier()

    # Load Feather file (DREAMER format)

    df = feather.read_feather(feather_path)

    predictions = []
    for i, row in df.iterrows():
        pow_vector = [row[col] for col in pow_columns]
        result = classifier.predict(pow_vector)

        pred_row = {
            "index": i,
            "true_valence": row["valence"],
            "true_arousal": row["arousal"],
            "pred_valence": result["valence"],
            "pred_arousal": result["arousal"],
            "pred_label": result["label"],
            "confidence": result["confidence"],
            "timestamp": time.time(),
        }
        predictions.append(pred_row)

    # Save predictions to CSV
    df_pred = pd.DataFrame(predictions)
    if os.path.exists(f"{OUTPUT_DIR}/predictions.csv"):
        output_csv = Path(f"{OUTPUT_DIR}/predictions{str(int(time.time()))}.csv")
    else:
        output_csv = Path(f"{OUTPUT_DIR}/predictions.csv")
    df_pred.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


def run_csv_mode(csv_path: str, output_csv: str = None):
    """
    Process L1 output from CSV.

    Args:
        csv_path: Path to L1 output CSV
        output_csv: Optional path to save L2 predictions (default: L2_predictions.csv)
    """
    print(f"Running L2 in CSV mode: {csv_path}")

    # Initialize classifier and consumer
    classifier = VAClassifier()
    consumer = CSVConsumer(csv_path)

    # Get rows with metadata
    rows = consumer.get_rows_with_metadata()
    print(f"Processing {len(rows)} rows...")

    # Predict on each row
    predictions = []
    for i, row in enumerate(rows):
        result = classifier.predict(row["pow_vector"])
        pred_row = {
            "index": i,
            "true_emot_state": row["emot_state"],
            "true_valence": row["valence"],
            "true_arousal": row["arousal"],
            "pred_valence": result["valence"],
            "pred_arousal": result["arousal"],
            "pred_label": result["label"],
            "confidence": result["confidence"],
            "smoothed": row["smoothed"],
            "timestamp": row["timestamp"],
        }
        predictions.append(pred_row)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} / {len(rows)}")

    # Save predictions to CSV
    if output_csv is None:
        output_csv = "L2_predictions.csv"

    import pandas as pd

    df_pred = pd.DataFrame(predictions)
    df_pred.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Mean pred valence: {df_pred['pred_valence'].mean():.3f}")
    print(f"Mean pred arousal: {df_pred['pred_arousal'].mean():.3f}")
    print(f"Mean confidence: {df_pred['confidence'].mean():.3f}")
    print(f"Label distribution:\n{df_pred['pred_label'].value_counts()}")


def run_queue_mode(l1_queue, duration_sec: int = 10, output_json: str = None):
    """
    Process live L1 output from queue.

    Args:
        l1_queue: The queue.Queue from L1 (EmotionSimulator)
        duration_sec: How long to process (seconds)
        output_json: Optional path to save predictions (default: L2_stream.jsonl)
    """
    print(f"Running L2 in queue mode for {duration_sec} seconds...")

    # Initialize classifier and consumer
    classifier = VAClassifier()
    consumer = QueueConsumer(l1_queue, timeout=0.5)

    predictions = []
    start_time = time.time()
    count = 0

    # Stream predictions
    if output_json is None:
        output_json = "L2_stream.jsonl"

    with open(output_json, "w") as f:
        for pow_vec in consumer.stream():
            result = classifier.predict(pow_vec)

            # Format output message
            msg = {
                "source": "L2_classifier",
                "count": count,
                "valence": result["valence"],
                "arousal": result["arousal"],
                "label": result["label"],
                "confidence": result["confidence"],
                "timestamp": time.time(),
            }

            predictions.append(msg)
            f.write(json.dumps(msg) + "\n")
            f.flush()

            count += 1
            if count % 10 == 0:
                print(f"  Processed {count} vectors")

            # Check if duration exceeded
            if time.time() - start_time > duration_sec:
                break

    print(f"Saved {count} predictions to {output_json}")


if __name__ == "__main__":
    config = load_yml_config()

    # Run in Feather mode (DREAMER dataset)
    run_feather_mode(config["feather_path"], config["POW_COLUMNS"])

    # Run in CSV mode (L1 output CSV)
    # run_csv_mode("L1_output.csv")

    # Run in queue mode (live L1 output)
    # run_queue_mode(l1_out, duration_sec=30, output_json="L2_live_predictions.jsonl")
