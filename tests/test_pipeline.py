"""End-to-end pipeline test with synthetic image."""

from pathlib import Path

from src.config import DemoConfig
from src.pipeline import ChestXrayPipeline


def test_pipeline_dry_run(tmp_path: Path):
    """Run full pipeline without local image; should fall back to synthetic."""
    config = DemoConfig(
        local_image_path=tmp_path / "missing_input.png",
        output_dir=tmp_path / "output",
    )
    pipeline = ChestXrayPipeline(config)
    pipeline.run()

    assert (config.output_dir / config.viz_filename).exists()
    assert (config.output_dir / "pipeline.log").exists()
