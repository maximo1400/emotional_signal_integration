"""
Layer 2 Runner: Example of how to use the classifier with L1 output

Modes:
1. feather_mode: Process L1 output from Feather file (DREAMER dataset)
2. queue_mode: Connect to live L1 output queue (real-time)
3. csv_mode: Process L1 output CSV (batch)
"""

import os
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


def _set_va_range(values: list[int], num_classes: int) -> list[int]:
    """Compress valence/arousal values to fit the number of classes.

    DREAMER valence/arousal are on a 1..5 scale. This maps that source scale
    into `num_classes` bins while keeping the midpoint stable for num_classes = 3.

    returns a list of integers in the range [0, num_classes-1] representing the compressed class indices.
    """
    if num_classes == 1:
        return [0] * len(values)

    if num_classes == 5:
        return [value - 1 for value in values]

    mapped_values = []
    for value in values:
        # Map the 1..5 range to 0..(num_classes-1) while the midpoint stays stable for  num_classes = 3
        class_index = round((float(value) - 1) * (num_classes - 1) / 4)
        class_index = max(0, min(num_classes - 1, class_index))
        mapped_values.append(class_index)

    return mapped_values


def _va_to_label(val: list, ar: list) -> list[str]:
    """Define the emotion label based on valence and arousal values and the defined emotional state areas."""
    labels_str = []
    for v, a in zip(val, ar):
        label = f"{v}_{a}"
        labels_str.append(label)
    return labels_str


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

    valence = _set_va_range(list(df["valence"]), num_classes)
    arousal = _set_va_range(list(df["arousal"]), num_classes)
    labels = _va_to_label(valence, arousal)

    pow_vectors = []
    df_pow = df.reindex(columns=pow_columns)
    for i, row in df_pow.iterrows():
        pow_vector = feat_select.process_data(row.tolist())
        pow_vectors.append(pow_vector)

    return pow_vectors, labels


def train_model(
    feather_path: str,
    pow_columns: list,
    output_dir: str,
    emotional_states_areas: list,
    classifier: str,
    classifier_hyperparameters: dict,
    num_classes: int,
    model_folder: str = None,
    models_names: dict[str, str] = None,
    # save_model_path: str = None,
):
    """
    Process from Feather file (DREAMER).

    Args:
        feather_path: Path to L1 output Feather file
        pow_columns: List of column names for power band features in the Feather file
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
    df = df.head(20)
    pow_data, labels = _collect_pow_features(df, pow_columns, num_classes)

    classifier_manager = ClassifierManager(pow_columns, emotional_states_areas)

    model_path = f"{model_folder}/{models_names[classifier]}"

    if os.path.exists(model_path):
        # Load/select existing model
        classifier_manager.select(
            classifier,
            model_path=model_path,
            hyperparams=classifier_hyperparameters,
            num_classes=num_classes,
            # label_va_lookup=label_va_lookup,
        )

        # Inspect saved model metadata and warn on mismatches with current inputs/outputs
        # TODO: Implement metadata saving and inspection in ClassifierManager
        # loaded_metadata = classifier_manager.get_model_metadata(model_path)
        # model_input_len = loaded_metadata["pow_columns"]
        # model_num_classes = loaded_metadata["num_classes"]
        # if model_input_len != len(pow_columns) or model_num_classes != num_classes:
        #     warnings.warn(
        #         f"Loaded model metadata mismatch: expected input length {model_input_len} and num_classes {model_num_classes}\
        #               but got input length {len(pow_columns)} and num_classes {num_classes}. Predictions may be unreliable."
        #     )

    else:
        classifier_manager.train(
            classifier,
            pow_data,
            labels,
            model_path=model_path,
            hyperparams=classifier_hyperparameters,
            num_classes=num_classes,
            # label_va_lookup=label_va_lookup,
        )


# def predict():
#     for i, row in enumerate(train_vectors):
#         pow_vector = row
#         result = classifier_manager.predict(pow_vector)

#         pred_row = {
#             "index": i,
#             "true_valence": row["valence"],
#             "true_arousal": row["arousal"],
#             "pred_valence": result["valence"],
#             "pred_arousal": result["arousal"],
#             "pred_label": result["label"],
#             "confidence": result["confidence"],
#             "timestamp": time.time(),
#         }
#         predictions.append(pred_row)

#     # Save predictions to CSV
#     df_pred = pd.DataFrame(predictions)

#     output_csv = run_output_dir / "predictions.csv"
#     features_csv = run_output_dir / "features.csv"

#     df_pred.to_csv(output_csv, index=False)
#     df_features = pd.DataFrame(feat_select.pow, columns=feat_select.labels)
#     df_features.to_csv(features_csv, index=False)
#     print(f"Predictions saved to {output_csv}")
#     print(f"Features saved to {features_csv}")
#     if not model_folder:
#         print(f"Trained classifier saved to {model_path}")


if __name__ == "__main__":
    config = get_config(
        [
            "feather_file_path",
            "POW_COLUMNS",
            "l2_output_folder",
            "emotional_states_areas",
            "classifier",
            "clasfier_mode",
            "classifier_hyperparameters",
            "num_classes",
            "models_folder",
            "models_names",
        ]
    )

    # Run in Feather mode (DREAMER dataset)
    if config["clasfier_mode"] == "train":
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
