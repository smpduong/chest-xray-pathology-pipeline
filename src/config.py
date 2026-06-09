"""Immutable configuration for the chest X-ray pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemoConfig:
    """Pipeline configuration container."""

    local_image_path: Path = Path("chest_xray_input.png")

    fallback_url: str = (
        "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/"
        "master/images/000001-12.jpg"
    )

    output_dir: Path = Path("./output")
    image_filename: str = "sample_chest_xray.png"
    viz_filename: str = "xray_pathologies.png"

    device: str = "cpu"
    request_timeout: int = 30
    prob_threshold: float = 0.5
