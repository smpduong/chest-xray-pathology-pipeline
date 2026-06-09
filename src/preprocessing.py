"""Radiological preprocessing for XRV-compatible inputs."""

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import torchxrayvision as xrv


def build_transforms() -> transforms.Compose:
    """XRV geometric transforms operating on numpy arrays."""
    return transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])


def preprocess(image: Image.Image, device: str) -> torch.Tensor:
    """Convert PIL grayscale to XRV tensor: (1, 1, 224, 224)."""
    img_np = np.array(image)  # (H, W), uint8
    img_np = xrv.datasets.normalize(img_np, 255)  # [-1024, 1024]
    img_np = img_np[None, ...]  # (1, H, W)

    geo_transform = build_transforms()
    img_np = geo_transform(img_np)  # (1, 224, 224)

    tensor = torch.from_numpy(img_np).unsqueeze(0).float()  # (1, 1, 224, 224)
    tensor = tensor.to(device, non_blocking=False)
    return tensor
