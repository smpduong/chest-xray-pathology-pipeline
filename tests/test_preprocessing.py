"""Test radiological preprocessing."""

import torch
from PIL import Image

from src.config import DemoConfig
from src.preprocessing import preprocess


def test_tensor_shape_and_channels():
    """XRV expects single-channel grayscale input."""
    config = DemoConfig()
    image = Image.new("L", (512, 512), color=128)
    tensor = preprocess(image, config.device)

    assert tensor.dim() == 4
    assert tensor.shape == (1, 1, 224, 224)
    assert tensor.dtype == torch.float32


def test_tensor_range():
    """Normalized pixel values should span the radiological domain."""
    config = DemoConfig()
    image = Image.new("L", (512, 512), color=255)
    tensor = preprocess(image, config.device)

    assert tensor.max().item() < 1025
    assert tensor.min().item() > -1025
