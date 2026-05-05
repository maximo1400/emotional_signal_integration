"""
Layer 2 Runner: Example of how to use the classifier with L1 output

Modes:
1. queue_mode: Connect to live L1 output queue (real-time)
2. csv_mode: Process L1 output CSV (batch)
"""

import sys
import json
import time
from pathlib import Path

from classifier import VAClassifier
from consumer import QueueConsumer, CSVConsumer


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
    """
    Usage examples:
    
    1. CSV mode (batch):
       python run.py --mode csv --csv L1_output.csv
    
    2. Queue mode (real-time, requires L1 running):
       python run.py --mode queue --duration 30
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Layer 2: Emotion State Estimation")
    parser.add_argument("--mode", choices=["csv", "queue"], default="csv",
                        help="Input mode: csv or queue")
    parser.add_argument("--csv", type=str, default="L1_output.csv",
                        help="Path to L1 output CSV (for csv mode)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save predictions")
    parser.add_argument("--duration", type=int, default=10,
                        help="Duration to process (seconds, for queue mode)")
    
    args = parser.parse_args()
    
    if args.mode == "csv":
        run_csv_mode(args.csv, args.output)
    elif args.mode == "queue":
        print("Queue mode requires a running L1 simulator with queue access.")
        print("This is a skeleton; implement integration with your main.py")
        raise NotImplementedError("Queue mode requires main.py orchestration")
