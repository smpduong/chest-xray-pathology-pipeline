"""Pathology visualization utilities."""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def visualize(
    image: Image.Image,
    predictions: Dict[str, float],
    threshold: float,
    output_path: Path,
) -> Path:
    """Side-by-side X-ray and pathology bar chart."""
    sorted_preds = sorted(predictions.items(), key=lambda kv: kv[1], reverse=True)
    top_k = 10
    top_preds = sorted_preds[:top_k]
    labels, values = zip(*top_preds) if top_preds else ([], [])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Chest X-ray Input")
    axes[0].axis("off")

    colors = ["crimson" if v > threshold else "steelblue" for v in values]
    y_pos = np.arange(len(labels))

    axes[1].barh(y_pos, values, color=colors, edgecolor="black", alpha=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1.0)
    axes[1].axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold})",
    )
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_title("Top Pathology Predictions (TorchXRayVision)")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
