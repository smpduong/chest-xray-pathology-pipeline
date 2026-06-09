"""End-to-end chest X-ray pathology pipeline."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import requests
import torch
from PIL import Image, UnidentifiedImageError

import torchxrayvision as xrv

from .config import DemoConfig
from .preprocessing import preprocess
from .utils import setup_logging
from .visualization import visualize


class ChestXrayPipeline:
    """Production-oriented XRV pipeline."""

    def __init__(self, config: DemoConfig) -> None:
        self.cfg = config
        self.logger = setup_logging(str(config.output_dir / "pipeline.log"))
        self.device = torch.device(config.device)
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Loading DenseNet121 from torchxrayvision...")
        try:
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        except Exception as exc:
            self.logger.critical("Failed to load XRV weights: %s", exc)
            raise RuntimeError(
                "Model load failed. Check internet, disk space, and "
                "~/.torchxrayvision/models_data/ for corrupted .pt files."
            ) from exc

        self.model.to(self.device)
        self.model.eval()

        self.pathologies: List[str] = self.model.pathologies
        self.logger.info(
            "Model loaded on %s | Pathologies: %d",
            self.device,
            len(self.pathologies),
        )

    def _download(self, url: str, destination: Path) -> bool:
        """Attempt single URL download. Return True on success."""
        self.logger.info("Attempting download: %s", url)
        try:
            response = requests.get(url, stream=True, timeout=self.cfg.request_timeout)
            response.raise_for_status()
        except Exception as exc:
            self.logger.warning("  -> Download failed: %s", exc)
            return False

        try:
            with open(destination, "wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
        except OSError as exc:
            self.logger.error("  -> Disk write error: %s", exc)
            return False

        self.logger.info("  -> Download complete: %s", destination)
        return True

    def acquire_image(self) -> Path:
        """Priority: local file -> download -> synthetic."""
        local = self.cfg.local_image_path
        if local.exists():
            self.logger.info("Using local image: %s", local.resolve())
            return local

        self.logger.info("Local image not found. Trying fallback download...")
        dest = self.cfg.output_dir / self.cfg.image_filename
        if self._download(self.cfg.fallback_url, dest):
            return dest

        self.logger.warning("Fallback failed. Generating synthetic image.")
        return self._generate_synthetic(dest)

    def _generate_synthetic(self, destination: Path) -> Path:
        """Generate synthetic bilateral lung X-ray."""
        self.logger.info("Generating synthetic chest X-ray...")
        h, w = 512, 512
        img = np.ones((h, w), dtype=np.uint8) * 200
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        mask_left = ((x - (cx - 80)) ** 2 / 60 ** 2 + (y - cy) ** 2 / 120 ** 2) <= 1
        img[mask_left] = 60
        mask_right = ((x - (cx + 80)) ** 2 / 60 ** 2 + (y - cy) ** 2 / 120 ** 2) <= 1
        img[mask_right] = 60
        noise = np.random.normal(0, 15, (h, w)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(destination)
        self.logger.info("Synthetic image saved: %s", destination)
        return destination

    def load_image(self, path: Path) -> Image.Image:
        """Open and validate grayscale image."""
        try:
            image = Image.open(path).convert("L")
        except UnidentifiedImageError as exc:
            self.logger.error("Invalid image file: %s", path)
            raise ValueError(f"Invalid image file: {path}") from exc
        except OSError as exc:
            self.logger.error("OS error opening image: %s", exc)
            raise

        self.logger.info(
            "Loaded image: %s | Mode: %s | Size: %s", path, image.mode, image.size
        )
        return image

    @torch.inference_mode()
    def run_inference(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Multi-label inference. Returns pathology probabilities."""
        if tensor.dim() != 4 or tensor.shape[1] != 1:
            raise ValueError(f"Expected (N, 1, H, W), got {tensor.shape}")

        outputs = self.model(tensor)
        probs = torch.sigmoid(outputs).squeeze(0)
        return {
            pathology: float(prob)
            for pathology, prob in zip(self.pathologies, probs)
        }

    def run(self) -> None:
        """Execute full pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("Starting Chest X-ray Pipeline (TorchXRayVision)")
        self.logger.info("=" * 60)

        img_path = self.acquire_image()
        image = self.load_image(img_path)
        input_tensor = preprocess(image, self.device)

        self.logger.info(
            "Input tensor: %s | Device: %s | Range: [%.1f, %.1f]",
            input_tensor.shape,
            input_tensor.device,
            input_tensor.min().item(),
            input_tensor.max().item(),
        )

        predictions = self.run_inference(input_tensor)
        flagged = {k: v for k, v in predictions.items() if v > self.cfg.prob_threshold}

        self.logger.info("-" * 60)
        self.logger.info("MULTI-LABEL PREDICTIONS (NON-DIAGNOSTIC)")
        self.logger.info("-" * 60)

        for pathology, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            marker = " 🚨" if prob > self.cfg.prob_threshold else ""
            self.logger.info("  %-35s %.3f%s", pathology + ":", prob, marker)

        self.logger.info("-" * 60)
        if flagged:
            self.logger.info(
                "Flagged (p > %.2f): %s", self.cfg.prob_threshold, ", ".join(flagged.keys())
            )
        else:
            self.logger.info("No pathologies exceeded threshold (p > %.2f).", self.cfg.prob_threshold)

        self.logger.info(
            "⚠️  DISCLAIMER: Research-grade probabilities only. NOT a medical diagnosis."
        )
        self.logger.info("-" * 60)

        viz_path = self.cfg.output_dir / self.cfg.viz_filename
        visualize(image, predictions, self.cfg.prob_threshold, viz_path)
        self.logger.info("Saved visualization: %s", viz_path)

        self.logger.info("Pipeline completed successfully.")
        self.logger.info("Output directory: %s", self.cfg.output_dir.resolve())
