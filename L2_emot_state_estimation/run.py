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
import queue
import pandas as pd

from L2_emot_state_estimation.EpochNormalizer import EPOCCrossSessionNormalizer
from L2_emot_state_estimation.classifier import ClassifierManager
from L2_emot_state_estimation.featureSelection import FeatureSelector
from L2_emot_state_estimation.utils import (
    PredictionWriter,
    _evaluate_predictions,
)

# Add parent directory to path to import config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config


def _row_to_pow_values(row, pow_columns: list[str]) -> list[float]:
    if isinstance(row, dict):
        row = row["pow"]

    if isinstance(row, pd.Series):
        return row.reindex(pow_columns).tolist()

    return list(row)


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
    normalizer: EPOCCrossSessionNormalizer,
) -> list[list[float]]:
    feat_select = FeatureSelector()
    df_pow = df.reindex(columns=pow_columns)

    rows = df_pow.to_numpy()
    rows = normalizer.process_batch(rows)

    return feat_select.process_data_batch(rows)


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
    normalizer = EPOCCrossSessionNormalizer(is_epoch_data=False)
    pow_vectors = _process_pow_vectors(df, pow_columns, normalizer)
    return pow_vectors, labels, people


def _build_output_file(
    output_csv_folder: str | Path,
    pow_data_source: str,
    filename: str = "predictions.csv",
    create_dir: bool = True,
) -> Path:
    run_stamp = str(int(time.time()))
    output_folder = Path(output_csv_folder) / f"{pow_data_source}_{run_stamp}"
    if create_dir:
        output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder / filename


def train_model():
    config = get_config([
        "feather_file_path",
        "POW_COLUMNS",
        "classifier",
        "classifier_hyperparameters",
        "num_classes",
        "models_folder",
        "models_names",
        "l2_output_folder",
        "class_balancing",
        "data_split_method",
        "data_split_parameters",
    ])

    classifier = config["classifier"]
    feather_path = config["feather_file_path"]

    model_folder = config["models_folder"]
    model_filename = config["models_names"][classifier]
    model_path = Path(model_folder) / model_filename

    print(f"Running L2 in train mode, input: {feather_path}")

    pow_vectors, labels, people = _collect_training_data(
        feather_path,
        config["POW_COLUMNS"],
        config["num_classes"],
    )

    classifier_input_len = len(pow_vectors[0])
    classifier_manager = ClassifierManager(classifier_input_len)

    train_result = classifier_manager.train(
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

    true_labels = [str(label) for label in train_result["y_test"]]
    predicted_labels = [str(label) for label in train_result["y_pred"]]

    output_file = _build_output_file(
        config["l2_output_folder"],
        "train",
        filename="test_predictions.csv",
    )

    eval_data = _evaluate_predictions(
        true_labels, predicted_labels, output_dir=output_file.parent, save_png=True
    )

    output_frame = pd.DataFrame({
        "true_label": true_labels,
        "pred_label": predicted_labels,
        "true_valence": eval_data["true_valence"],
        "true_arousal": eval_data["true_arousal"],
        "pred_valence": eval_data["pred_valence"],
        "pred_arousal": eval_data["pred_arousal"],
    })

    output_frame.to_csv(output_file, index=False)

    report_file = output_file.with_name("evaluation.txt")
    report_file.write_text("\n".join(eval_data["formatted_report"]), encoding="utf-8")

    print(f"Saved train/test comparison to {output_file}")
    print(f"Saved evaluation report to {report_file}")


def predict_from_file(l1_queue: queue.Queue, l2_queue: queue.Queue):
    config = get_config([
        "POW_COLUMNS",
        "num_classes",
        "predict_from_file_pow_csv_path",
    ])

    pow_csv_path = Path(config["predict_from_file_pow_csv_path"])

    if not pow_csv_path.exists():
        raise FileNotFoundError(f"Missing pow.csv at {pow_csv_path}")

    print(f"Running L2 in predict_from_file mode: {pow_csv_path}")
    df = pd.read_csv(pow_csv_path)
    df.columns = [str(column).strip() for column in df.columns]

    true_labels = None
    if "valence" in df.columns and "arousal" in df.columns:
        num_classes = config["num_classes"]
        true_valences = _set_va_range(df["valence"].tolist(), num_classes)
        true_arousals = _set_va_range(df["arousal"].tolist(), num_classes)
        true_labels = _va_to_label(true_valences, true_arousals)

    df_pow = df.reindex(columns=config["POW_COLUMNS"])
    for row in df_pow.itertuples(index=False, name=None):
        l1_queue.put(list(row))
    l1_queue.put(None)

    print(
        "Finished sending pow vectors to L1 queue, now will run L2 in predict_from_queue mode"
    )
    predict_from_queue(l1_queue, l2_queue, true_labels=true_labels)


def predict_from_queue(
    l1_queue: queue.Queue, l2_queue: queue.Queue, true_labels: list[str] = None
):
    config = get_config([
        "models_folder",
        "classifier",
        "POW_COLUMNS",
        "num_classes",
        "models_names",
        "l2_output_folder",
        "pow_data_source",
        "verbose",
        "save_output_files",
    ])
    verbose = config["verbose"]
    classifier = config["classifier"]
    pow_columns = config["POW_COLUMNS"]
    save_files = config["save_output_files"]
    num_classes = config["num_classes"]

    model_folder = config["models_folder"]
    model_filename = config["models_names"][classifier]
    model_path = Path(model_folder) / model_filename

    print("Running L2 in predict_from_queue mode")

    feat_select = FeatureSelector()

    output_file = _build_output_file(
        config["l2_output_folder"],
        config["pow_data_source"],
        filename="queue_predictions.csv",
        create_dir=save_files,
    )

    pred_writer = PredictionWriter(output_file, save_files, true_labels)
    predicted_labels = []

    try:
        first_loop = True
        is_epoch_data = config["pow_data_source"] == "emotiv"
        normalizer = EPOCCrossSessionNormalizer(is_epoch_data=is_epoch_data)

        while True:
            row = l1_queue.get()

            if row is None:
                if verbose:
                    print("L2 queue received stop signal")
                l2_queue.put(None)  # Signal to L3 that predictions are done
                break

            pow_values = _row_to_pow_values(row, pow_columns)
            normalized_values = normalizer.new_row(pow_values)
            pow_row = feat_select.process_data(normalized_values)

            if first_loop:
                pred_writer.write_headers(feat_select.get_final_feature_names())
                classifier_manager = ClassifierManager(len(pow_row))
                classifier_manager.load(model_path, num_classes=num_classes)
                first_loop = False

            prediction = classifier_manager.predict_with_confidence(pow_row)
            prediction_label = prediction["label"]
            prediction_conf = prediction["confidence"]
            timestamp = time.time()

            predicted_labels.append(str(prediction["label"]))

            pred_writer.write_row(pow_row, prediction_label, prediction_conf, timestamp)

            payload = {
                "label": prediction_label,
                "confidence": prediction_conf,
                "timestamp": timestamp,
            }
            l2_queue.put(payload)
            print(f"Classifier output: {payload}")
    finally:
        pred_writer.close()
        if verbose and save_files:
            print(f"Saved queue predictions to {output_file}")

        if true_labels is not None and len(true_labels) == len(predicted_labels):
            print("\nEvaluating Virtual File Predictions:")
            if save_files:
                report = _evaluate_predictions(
                    true_labels,
                    predicted_labels,
                    output_dir=output_file.parent,
                    save_png=save_files,
                )
                report_file = output_file.with_name("evaluation.txt")
                report_file.write_text(
                    "\n".join(report["formatted_report"]), encoding="utf-8"
                )

                print(f"Saved evaluation report to {report_file}")
            else:
                report = _evaluate_predictions(true_labels, predicted_labels)
            l2_queue.put(None)


def run_l2(l1_queue: queue.Queue = None, l2_out_queue: queue.Queue = None):
    config = get_config(["classifier_mode"])
    mode = config["classifier_mode"]

    if mode == "train":
        train_model()
    elif mode == "predict_from_file":
        predict_from_file(l1_queue, l2_out_queue)
    elif mode == "predict_from_queue":
        predict_from_queue(l1_queue, l2_out_queue)
