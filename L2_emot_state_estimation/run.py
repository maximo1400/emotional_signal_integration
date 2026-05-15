"""
Layer 2 runner.

Modes:
- train: train the classifier using processed DREAMER data
- predict_from_file: load a trained model and predict from an L1 pow.csv file
- predict_from_queue: load a trained model from file and use it to make
  predictions on incoming data from the L1 queue
"""

import sys
import time
from pathlib import Path

import pandas as pd

from classifier import ClassifierManager
from featureSelection import FeatureSelector
from sklearn.metrics import accuracy_score, classification_report
from utils import plot_confusion_matrix

# Add parent directory to path to import config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config


def _set_va_range(values: list[int], num_classes: int) -> list[int]:
    """Compress DREAMER 1..5 valence/arousal values into 0..num_classes-1."""
    if num_classes == 1:
        return [0] * len(values)

    if num_classes == 5:
        return [value - 1 for value in values]

    # Map 1..5 to 0..(num_classes-1), keeping midpoint stable for num_classes=3
    return [
        max(
            0,
            min(num_classes - 1, round((float(value) - 1) * (num_classes - 1) / 4)),
        )
        for value in values
    ]


def _va_to_label(val: list[int], ar: list[int]) -> list[str]:
    return [f"{v}_{a}" for v, a in zip(val, ar)]


def _process_pow_vectors(
    df: pd.DataFrame,
    pow_columns: list[str],
) -> list[list[float]]:
    feat_select = FeatureSelector()
    df_pow = df.reindex(columns=pow_columns)

    return [
        feat_select.process_data(list(row))
        for row in df_pow.itertuples(index=False, name=None)
    ]


def _collect_training_data(
    feather_path: str | Path,
    pow_columns: list[str],
    num_classes: int,
) -> tuple[list[list[float]], list[str], list[int]]:
    df = pd.read_feather(feather_path)

    valence = _set_va_range(df["valence"].tolist(), num_classes)
    arousal = _set_va_range(df["arousal"].tolist(), num_classes)
    labels = _va_to_label(valence, arousal)
    people = df["subject_id"].tolist()
    pow_vectors = _process_pow_vectors(df, pow_columns)

    return pow_vectors, labels, people


def _build_model_path(
    models_folder: str | Path,
    models_names: dict[str, str],
    classifier: str,
) -> Path:
    return Path(models_folder) / models_names[classifier]


def _build_output_file(
    output_csv_folder: str | Path,
    pow_data_source: str,
    filename: str = "predictions.csv",
) -> Path:
    run_stamp = str(int(time.time()))
    output_folder = Path(output_csv_folder) / f"{pow_data_source}_{run_stamp}"
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder / filename


def train_model():
    config = get_config(
        [
            "feather_file_path",
            "POW_COLUMNS",
            "classifier",
            "classifier_hyperparameters",
            "num_classes",
            "models_folder",
            "models_names",
            "class_balancing",
            "data_split_method",
            "data_split_parameters",
        ]
    )

    classifier = config["classifier"]
    feather_path = config["feather_file_path"]

    print(f"Running L2 in train mode: {feather_path}")

    pow_vectors, labels, people = _collect_training_data(
        feather_path,
        config["POW_COLUMNS"],
        config["num_classes"],
    )

    classifier_input_len = len(pow_vectors[0])
    classifier_manager = ClassifierManager(classifier_input_len)

    model_path = _build_model_path(
        config["models_folder"],
        config["models_names"],
        classifier,
    )

    classifier_manager.train(
        classifier,
        pow_vectors,
        labels,
        model_path,
        config["classifier_hyperparameters"][classifier],
        config["num_classes"],
        config["class_balancing"],
        config["data_split_method"],
        config["data_split_parameters"],
        people,
    )


def predict_from_file():
    config = get_config(
        [
            "models_folder",
            "classifier",
            "pow_data_source",
            "l2_output_folder",
            "POW_COLUMNS",
            "num_classes",
            "models_names",
            "predict_from_file_pow_csv_path",
        ]
    )

    classifier = config["classifier"]
    pow_csv_path = Path(config["predict_from_file_pow_csv_path"])

    if not pow_csv_path.exists():
        raise FileNotFoundError(f"Missing pow.csv at {pow_csv_path}")

    print(f"Running L2 in predict_from_file mode: {pow_csv_path}")

    df = pd.read_csv(pow_csv_path)
    df.columns = [str(column).strip() for column in df.columns]

    pow_vectors = _process_pow_vectors(df, config["POW_COLUMNS"])

    classifier_input_len = len(pow_vectors[0])
    classifier_manager = ClassifierManager(classifier_input_len)

    model_path = _build_model_path(
        config["models_folder"],
        config["models_names"],
        classifier,
    )
    classifier_manager.load(model_path, num_classes=config["num_classes"])

    predictions = classifier_manager.batch_predict_with_confidence(pow_vectors)
    output_frame = pd.DataFrame(predictions)

    if "valence" in df.columns and "arousal" in df.columns:
        true_valence = _set_va_range(df["valence"].tolist(), config["num_classes"])
        true_arousal = _set_va_range(df["arousal"].tolist(), config["num_classes"])
        true_labels = _va_to_label(true_valence, true_arousal)

        predicted_valence = []
        predicted_arousal = []
        predicted_labels = []
        for prediction in predictions:
            label = prediction["label"]
            predicted_labels.append(str(label))

            va, ar = label.split("_")
            predicted_valence.append(str(va))
            predicted_arousal.append(str(ar))

        print("Joint accuracy:", accuracy_score(true_labels, predicted_labels))
        print("Valence accuracy:", accuracy_score(true_valence, predicted_valence))
        print("Arousal accuracy:", accuracy_score(true_arousal, predicted_arousal))
        print(classification_report(true_labels, predicted_labels, zero_division=0))
        plot_confusion_matrix(true_labels, predicted_labels)

        output_frame["true_label"] = true_labels
        output_frame["true_valence"] = true_valence
        output_frame["true_arousal"] = true_arousal
        output_frame["pred_valence"] = predicted_valence
        output_frame["pred_arousal"] = predicted_arousal

    output_file = _build_output_file(
        config["l2_output_folder"],
        config["pow_data_source"],
    )
    output_frame.to_csv(output_file, index=False)

    print(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    config = get_config(["classifier_mode"])

    if config["classifier_mode"] == "train":
        train_model()

    elif config["classifier_mode"] == "predict_from_file":
        predict_from_file()

    elif config["classifier_mode"] == "predict_from_queue":
        raise Warning(
            "predict_from_queue needs to be runned from main.py to access the L1 queue"
        )
