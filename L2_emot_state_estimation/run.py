"""
Layer 2 Runner: Example of how to use the classifier with L1 output

Modes:
1. feather_mode: Process L1 output from Feather file (DREAMER dataset)
2. queue_mode: Connect to live L1 output queue (real-time)
3. csv_mode: Process L1 output CSV (batch)
"""

import sys
import time
from pathlib import Path
import pyarrow.feather as feather
import pandas as pd

from featureSelection import FeatureSelector
from classifier import ClassifierManager

# Add parent directory to path to import config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config


def _compress_va_range(list, num_classes):
    """Compress valence/arousal values to fit the number of classes.
    DREAMER has valence/arousal in [1, 5].
    Tries to mantain center values and compress/expand the range accordingly
    """
    if num_classes == 1:
        return [1 for x in list]
    elif num_classes == 3:
        return [2 if x == 3 else (1 if x < 3 else 3) for x in list]
    elif num_classes == 5:
        return list
    else:
        raise ValueError(
            "num_classes not implemented, need to implement compression/expansion logic for the given number of classes"
        )


def _define_label_va(val: list, ar: list):
    """Define the emotion label based on valence and arousal values and the defined emotional state areas."""
    max_label = max(max(val), max(ar))
    min_label = min(min(val), min(ar))
    labels_int = []
    labels_str = []
    i = 0


def _collect_pow_features(
    df: pd.DataFrame, pow_columns: list, num_classes: int
) -> tuple[
    list[list[float]],
    list[str],
    FeatureSelector,
    dict[str, dict[str, tuple[float, float]]],
]:
    """Extract processed power features and emotion labels from a labeled dataframe."""

    feat_select = FeatureSelector()
    pow_vectors = []
    labels = []
    label_va_samples = {}

    valence = list(df["valence"])
    arousal = list(df["arousal"])
    valence = _compress_va_range(valence, num_classes)
    arousal = _compress_va_range(arousal, num_classes)

    pow_df = df[pow_columns]

    for _, row in df.iterrows():
        label = row.get("emot_state")
        if pd.isna(label):
            continue

        label = str(label)
        pow_vector = [row[col] for col in pow_columns]
        processed_vector = feat_select.process_data(pow_vector)
        pow_vectors.append(processed_vector)
        labels.append(label)
        label_va_samples.setdefault(label, []).append(
            (float(row["valence"]), float(row["arousal"]))
        )

    label_va_lookup = {}
    for label, samples in label_va_samples.items():
        valence_mean = float(sum(sample[0] for sample in samples) / len(samples))
        arousal_mean = float(sum(sample[1] for sample in samples) / len(samples))
        label_va_lookup[label] = {
            "valence": (valence_mean, valence_mean),
            "arousal": (arousal_mean, arousal_mean),
        }

    return pow_vectors, labels, feat_select, label_va_lookup


def train_model(
    feather_path: str,
    pow_columns: list,
    output_dir: str,
    emotional_states_areas: list,
    classifier: str,
    classifier_hyperparameters: dict,
    num_classes: int,
    model_path: str = None,
    models_names: dict[str, str] = None,
    # save_model_path: str = None,
):
    """
    Process from Feather file (DREAMER).

    Args:
        feather_path: Path to L1 output Feather file
        pow_columns: List of column names for power features in the Feather file
        output_dir: Directory to save L2 predictions CSV
        emotional_states_areas: List of dicts defining emotional state areas in VA space
        classifier: Classifier type (e.g., "stub", "svm", "nn")
        classifier_hyperparameters: Hyperparameters for the classifier (dict)
        num_classes: Number of emotion classes to predict
    """
    print(f"Running L2 in Feather mode: {feather_path}")

    run_stamp = str(int(time.time()))
    run_output_dir = Path(output_dir) / run_stamp
    run_output_dir.mkdir(parents=True, exist_ok=True)

    df = feather.read_feather(feather_path)

    train_vectors, train_labels, _, label_va_lookup = _collect_pow_features(
        df, pow_columns
    )

    if not train_vectors:
        raise RuntimeError("No labeled rows were found for classifier training")

    classifier_manager = ClassifierManager(pow_columns, emotional_states_areas)
    if model_path:
        classifier_manager.select(
            classifier,
            model_path=model_path,
            hyperparams=classifier_hyperparameters,
            num_classes=num_classes,
            label_va_lookup=label_va_lookup,
        )
    else:
        classifier_manager.train(
            classifier,
            train_vectors,
            train_labels,
            model_path=save_model_path,
            hyperparams=classifier_hyperparameters,
            num_classes=num_classes,
            label_va_lookup=label_va_lookup,
        )

    predictions = []
    predict_df = df.head(20)
    feat_select = FeatureSelector()

    for i, row in predict_df.iterrows():
        pow_vector = [row[col] for col in pow_columns]
        pow_vector = feat_select.process_data(pow_vector)
        result = classifier_manager.predict(pow_vector)

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

    output_csv = run_output_dir / "predictions.csv"
    features_csv = run_output_dir / "features.csv"

    df_pred.to_csv(output_csv, index=False)
    df_features = pd.DataFrame(feat_select.pow, columns=feat_select.labels)
    df_features.to_csv(features_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    print(f"Features saved to {features_csv}")
    if not model_path:
        print(f"Trained classifier saved to {save_model_path}")


if __name__ == "__main__":
    config = get_config(
        [
            "feather_file_path",
            "POW_COLUMNS",
            "l2_output_folder",
            "emotional_states_areas",
            "classifier",
            "classifier_hyperparameters",
            "num_classes",
            "models_folder",
            "models_names",
        ]
    )

    # Run in Feather mode (DREAMER dataset)
    train_model(
        config["feather_file_path"],
        config["POW_COLUMNS"],
        config["l2_output_folder"],
        config["emotional_states_areas"],
        config["classifier"],
        config["classifier_hyperparameters"],
        config["num_classes"],
        config["models_folder"],
        config["models_names"],
    )
