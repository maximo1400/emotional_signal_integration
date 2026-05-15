"""
Layer 2 Runner: Example of how to use the classifier with L1 output

Modes:
train: Train the classifier using the processed data from DREAMER.
predict_from_queue: Load a trained model from file and use it to make predictions on incoming data from L1 queue.
predict_from_file: Load a trained model from file and use it to make predictions on newest L1 output data.
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
    people = df["subject_id"].to_list()

    valence = _set_va_range(list(df["valence"]), num_classes)
    arousal = _set_va_range(list(df["arousal"]), num_classes)
    labels = _va_to_label(valence, arousal)

    pow_vectors = []
    df_pow = df.reindex(columns=pow_columns)
    for i, row in df_pow.iterrows():
        pow_vector = feat_select.process_data(row.tolist())
        pow_vectors.append(pow_vector)

    return pow_vectors, labels, people


def train_model(
    feather_path: str,
    pow_columns: list,
    output_dir: str,
    classifier: str,
    classifier_hyperparameters: dict,
    num_classes: int,
    model_folder: str,
    models_names: dict[str, str],
    class_balancing: str,
    data_split_method: str,
    data_split_parameters: dict,
    pow_data_source: str,
):
    """
    Process from Feather file (DREAMER).

    Args:
        feather_path: Path to L1 output Feather file
        pow_columns: List of column names for power band features in the Feather file
        output_dir: Directory to save L2 predictions CSV
        classifier: Classifier type (e.g., "random_forest", "svm", "knn")
        classifier_hyperparameters: Hyperparameters for the classifier (dict)
        num_classes: Number of emotion classes to predict
        class_balancing: Method for class balancing ("none", "undersample", "oversample")
        data_split_method: Method for splitting data into train/test ("random", "subject")
        data_split_parameters: Parameters for the data splitting method (dict)
    """
    print(f"Running L2 in Feather mode: {feather_path}")

    df = feather.read_feather(feather_path)
    # df = df.head(5000)
    df = df[df["subject_id"] < 5]
    pow_data, labels, people = _collect_pow_features(df, pow_columns, num_classes)
    classifier_input_len = len(pow_data[0])

    classifier_manager = ClassifierManager(classifier_input_len)

    model_path = f"{model_folder}/{models_names[classifier]}"

    classifier_manager.train(
        classifier,
        pow_data,
        labels,
        model_path,
        classifier_hyperparameters[classifier],
        num_classes,
        class_balancing,
        data_split_method,
        data_split_parameters,
        people,
    )

    run_stamp = str(int(time.time()))
    output_folder = Path(f"{output_dir}/{pow_data_source}_{run_stamp}")
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "predictions.csv"


def predict_from_file(
    model_folder: str,
    classifier: str,
    l1_output_folder: str,
    pow_data_source: str,
    output_csv_folder: str,
    pow_columns: list,
    num_classes: int,
    models_names: dict[str, str],
    pow_csv_path: str,
):
    """
    Load a trained model from file and use it to make predictions on newest L1 output data.

    Args:
        model_path: Path to the saved model file (e.g., .joblib)
        l1_output_folder: Path to L1 output csv file
        pow_data_source: Source of power data in L1 output ("emotiv" or "virtual")
        output_csv_path: Path to save the predictions CSV
    """

    if not pow_csv_path.exists():
        raise FileNotFoundError(f"Missing pow.csv at {pow_csv_path}")

    df = pd.read_csv(pow_csv_path)
    df.columns = [str(column).strip() for column in df.columns]

    feat_select = FeatureSelector()
    pow_vectors = []
    df_pow = df.reindex(columns=pow_columns)

    for _, row in df_pow.iterrows():
        pow_vectors.append(feat_select.process_data(row.tolist()))

    classifier_input_len = len(pow_vectors[0])
    classifier_manager = ClassifierManager(classifier_input_len)

    model_folder = f"{model_folder}/{models_names[classifier]}"
    classifier_manager.load(str(model_folder), num_classes=num_classes)

    predictions = classifier_manager.batch_predict_with_confidence(pow_vectors)
    output_frame = pd.DataFrame(predictions)

    run_stamp = str(int(time.time()))
    output_folder = Path(f"{output_csv_folder}/{pow_data_source}_{run_stamp}")
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "predictions.csv"

    output_frame.to_csv(output_file, index=False)
    print(f"Running L2 in predict_from_file mode: {pow_csv_path}")
    print(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    config = get_config(
        [
            "feather_file_path",
            "POW_COLUMNS",
            "l2_output_folder",
            "classifier",
            "classifier_mode",
            "classifier_hyperparameters",
            "num_classes",
            "models_folder",
            "models_names",
            "class_balancing",
            "data_split_method",
            "data_split_parameters",
            "L1_output_folder",
            "pow_data_source",
            "predict_from_file_pow_csv_path",
        ]
    )

    # Run in Feather mode (DREAMER dataset)
    if config["classifier_mode"] == "train":
        train_model(
            config["feather_file_path"],
            config["POW_COLUMNS"],
            config["l2_output_folder"],
            config["classifier"],
            config["classifier_hyperparameters"],
            config["num_classes"],
            config["models_folder"],
            config["models_names"],
            config["class_balancing"],
            config["data_split_method"],
            config["data_split_parameters"],
            config["pow_data_source"],
        )

    elif config["classifier_mode"] == "predict_from_file":
        predict_from_file(
            config["models_folder"],
            config["classifier"],
            config["L1_output_folder"],
            config["pow_data_source"],
            config["l2_output_folder"],
            config["POW_COLUMNS"],
            config["num_classes"],
            config["models_names"],
            config["predict_from_file_pow_csv_path"],
        )
