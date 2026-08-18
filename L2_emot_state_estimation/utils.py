import csv
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    normalize=False,
    figsize=(10, 8),
    cmap="Blues",
    title=None,
    save_png=False,
    png_path="confusion_matrix.png",
    dpi=300,
):
    """
    Plot a styled confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Optional list of label names in desired order.
        normalize: If True, plot row-normalized values.
        figsize: Figure size as (width, height).
        cmap: Matplotlib colormap name.
        title: Optional plot title.
        save_png: If True, save the figure as a PNG.
        png_path: Output file path for PNG.
        dpi: Resolution for saved image.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_plot = np.nan_to_num(cm_plot)
        annot_fmt = ".2f"
        default_title = "Normalized Confusion Matrix"
    else:
        cm_plot = cm
        annot_fmt = "d"
        default_title = "Confusion Matrix"

    plt.figure(figsize=figsize)
    sns.set_theme(style="white", font_scale=1.1)

    ax = sns.heatmap(
        cm_plot,
        annot=True,
        fmt=annot_fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="lightgray",
        cbar=True,
        square=True,
        annot_kws={"size": 10},
    )

    ax.set_xlabel("Predicted label", fontsize=12, labelpad=12)
    ax.set_ylabel("True label", fontsize=12, labelpad=12)
    ax.set_title(title or default_title, fontsize=14, pad=16)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_png:
        plt.savefig(png_path, dpi=dpi, bbox_inches="tight")

    plt.close()


class PredictionWriter:
    def __init__(
        self, output_file: Path, save_files: bool, true_labels: list[str] | None = None
    ):
        self.save_files = save_files
        self.true_labels = true_labels
        self.predicted_count = 0
        self.f = None
        self.writer = None

        if self.save_files:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            self.f = output_file.open("w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.f)

    def write_headers(self, feature_names: list[str]):
        if not self.save_files or self.writer is None:
            return
        headers = feature_names + [
            "predicted_label",
            "confidence",
            "timestamp",
            "previous_layer_timestamp",
            "classifier_mode",
        ]
        if self.true_labels is not None:
            headers.append("y_true")
        self.writer.writerow(headers)

    def write_row(
        self,
        pow_vector: list[float],
        prediction_label: str,
        prediction_confidence: float,
        timestamp: float,
        previous_layer_timestamp: float,
        classifier_mode: str,
    ):
        if not self.save_files or self.writer is None or self.f is None:
            self.predicted_count += 1
            return

        row = [
            *pow_vector,
            prediction_label,
            prediction_confidence,
            timestamp,
            previous_layer_timestamp,
            classifier_mode,
        ]
        if self.true_labels is not None:
            row.append(self.true_labels[self.predicted_count])
        self.writer.writerow(row)
        self.f.flush()
        self.predicted_count += 1

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None
            self.writer = None


def _split_va_labels(labels: list[str] | pd.Series) -> tuple[list[str], list[str]]:
    valence = []
    arousal = []

    for label in labels:
        va, ar = str(label).split("-")
        valence.append(va)
        arousal.append(ar)

    return valence, arousal


def _evaluate_predictions(
    true_y: list[str],
    pred_y: list[str],
    output_dir: Path | str,
    save_png: bool = True,
    prefix: str = "",
) -> dict[str, list[str]]:
    true_val, true_ar = _split_va_labels(true_y)
    pred_val, pred_ar = _split_va_labels(pred_y)

    val_acc = accuracy_score(true_val, pred_val)
    ar_acc = accuracy_score(true_ar, pred_ar)
    joint_acc = accuracy_score(true_y, pred_y)

    val_prec = precision_score(true_val, pred_val, average="weighted", zero_division=0)
    val_rec = recall_score(true_val, pred_val, average="weighted", zero_division=0)
    val_f1 = f1_score(true_val, pred_val, average="weighted", zero_division=0)

    ar_prec = precision_score(true_ar, pred_ar, average="weighted", zero_division=0)
    ar_rec = recall_score(true_ar, pred_ar, average="weighted", zero_division=0)
    ar_f1 = f1_score(true_ar, pred_ar, average="weighted", zero_division=0)

    joint_prec = precision_score(true_y, pred_y, average="weighted", zero_division=0)
    joint_rec = recall_score(true_y, pred_y, average="weighted", zero_division=0)
    joint_f1 = f1_score(true_y, pred_y, average="weighted", zero_division=0)

    class_report = classification_report(true_y, pred_y, zero_division=0)
    valence_report = classification_report(true_val, pred_val, zero_division=0)
    arousal_report = classification_report(true_ar, pred_ar, zero_division=0)

    print("Valence accuracy:", val_acc)
    print("Arousal accuracy:", ar_acc)
    print("Joint accuracy:", joint_acc)
    print("classification_report:\n", class_report)

    png_path = Path(output_dir) / f"{prefix}confusion_matrix.png"
    plot_confusion_matrix(true_y, pred_y, save_png=save_png, png_path=str(png_path))

    formatted_report = [
        "Valence Metrics:",
        f"  Accuracy:  {val_acc:.4f}",
        f"  Precision: {val_prec:.4f}",
        f"  Recall:    {val_rec:.4f}",
        f"  F1 Score:  {val_f1:.4f}",
        "",
        "Arousal Metrics:",
        f"  Accuracy:  {ar_acc:.4f}",
        f"  Precision: {ar_prec:.4f}",
        f"  Recall:    {ar_rec:.4f}",
        f"  F1 Score:  {ar_f1:.4f}",
        "",
        "Joint Metrics:",
        f"  Accuracy:  {joint_acc:.4f}",
        f"  Precision: {joint_prec:.4f}",
        f"  Recall:    {joint_rec:.4f}",
        f"  F1 Score:  {joint_f1:.4f}",
        "",
        "Valence Report:\n",
        valence_report,
        "Arousal Report:\n",
        arousal_report,
        "Joint Report:\n",
        class_report,
    ]

    return {
        "true_valence": true_val,
        "true_arousal": true_ar,
        "pred_valence": pred_val,
        "pred_arousal": pred_ar,
        "valence_acc": val_acc,
        "arousal_acc": ar_acc,
        "Joint_accuracy": joint_acc,
        "valence_precision": val_prec,
        "valence_recall": val_rec,
        "valence_f1": val_f1,
        "arousal_precision": ar_prec,
        "arousal_recall": ar_rec,
        "arousal_f1": ar_f1,
        "Joint_precision": joint_prec,
        "Joint_recall": joint_rec,
        "Joint_f1": joint_f1,
        "classification_report": class_report,
        "valence_report": valence_report,
        "arousal_report": arousal_report,
        "formatted_report": formatted_report,
    }
