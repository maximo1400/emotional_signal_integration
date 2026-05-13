import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


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

    plt.show()
