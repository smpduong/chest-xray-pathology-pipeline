"""End-to-end pipeline test with synthetic image."""

from src.config import DemoConfig
from src.pipeline import ChestXrayPipeline


def test_pipeline_dry_run():
    """Run full pipeline without local image; should fall back to synthetic."""
    config = DemoConfig()
    pipeline = ChestXrayPipeline(config)

    if config.local_image_path.exists():
        config.local_image_path.unlink()

    pipeline.run()

    assert (config.output_dir / config.viz_filename).exists()
    assert (config.output_dir / "pipeline.log").exists()
