"""
Core inference logic for the satellite change-detection backend.
Adapted from the original Colab test script into reusable functions.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import SatelliteChangeNet


def read_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        array = np.load(path)
    elif suffix in {".tif", ".tiff"}:
        try:
            import rasterio
            with rasterio.open(path) as src:
                array = src.read()
        except Exception:
            with Image.open(path) as image:
                array = np.asarray(image)
    else:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))

    array = np.asarray(array).copy()

    if array.ndim == 2:
        array = array[..., None]
    elif array.ndim == 3:
        if array.shape[0] <= 32 and array.shape[1] > 32 and array.shape[2] > 32:
            array = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape {array.shape} in {path}")

    array = array.astype(np.float32, copy=False)

    finite = np.isfinite(array)
    array[~finite] = 0.0

    if array.max(initial=0.0) > 1.0:
        if array.max(initial=0.0) <= 255.0:
            array /= 255.0
        else:
            flat = array.reshape(-1, array.shape[-1])
            scale = np.percentile(flat, 99.5, axis=0)
            scale = np.maximum(scale, 1e-6).reshape(1, 1, -1)
            array = array / scale

    return np.clip(array, 0.0, 1.0)


def qa_proxy(image: torch.Tensor) -> torch.Tensor:
    channels = image.shape[0]
    rgb = image[: min(3, channels)]

    if rgb.shape[0] == 1:
        luminance = rgb[0]
        saturation = torch.zeros_like(luminance)
    else:
        luminance = (
            0.299 * rgb[0]
            + 0.587 * rgb[min(1, rgb.shape[0] - 1)]
            + 0.114 * rgb[min(2, rgb.shape[0] - 1)]
        )
        saturation = rgb.max(dim=0).values - rgb.min(dim=0).values

    possible_cloud = (luminance > 0.97) & (saturation < 0.12)
    possible_shadow = (luminance < 0.035) & (saturation < 0.20)
    invalid = (possible_cloud | possible_shadow).float().unsqueeze(0)

    invalid = F.max_pool2d(invalid.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]

    return 1.0 - invalid


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_config = checkpoint.get("model_config")
    else:
        state_dict = checkpoint
        config_path = Path(checkpoint_path).parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"{checkpoint_path} has no embedded model_config and no "
                f"config.json was found next to it."
            )
        with open(config_path) as f:
            model_config = json.load(f)["model_config"]

    model = SatelliteChangeNet(**(model_config or {})).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_pair(path_a, path_b, device):
    a = read_image(Path(path_a))
    b = read_image(Path(path_b))

    a_tensor = torch.from_numpy(np.ascontiguousarray(a)).float().permute(2, 0, 1)
    b_tensor = torch.from_numpy(np.ascontiguousarray(b)).float().permute(2, 0, 1)

    if b_tensor.shape[0] != a_tensor.shape[0]:
        raise ValueError(f"Channel mismatch between {path_a} and {path_b}")

    if b_tensor.shape[-2:] != a_tensor.shape[-2:]:
        b_tensor = F.interpolate(
            b_tensor.unsqueeze(0), size=a_tensor.shape[-2:],
            mode="bilinear", align_corners=False,
        )[0]

    qa_a = qa_proxy(a_tensor)
    qa_b = qa_proxy(b_tensor)

    x = torch.stack((a_tensor, b_tensor), dim=0).unsqueeze(0).to(device)
    qa = torch.stack((qa_a, qa_b), dim=0).unsqueeze(0).to(device)

    return x, qa, a_tensor.shape[-2:]


def pad_to_multiple(x, qa, multiple):
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple

    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
        qa = F.pad(qa, (0, pad_w, 0, pad_h), value=0.0)

    return x, qa, pad_h, pad_w


@torch.no_grad()
def run_whole_image(model, x, qa, orig_hw, multiple=32):
    x_padded, qa_padded, pad_h, pad_w = pad_to_multiple(x, qa, multiple)

    logits, ndvi_delta, ndwi_delta, ndvi_epochs, ndwi_epochs = model(
        x_padded, qa_padded, return_indices=True
    )

    probability = torch.sigmoid(logits)[0, 0]
    ndvi_delta = ndvi_delta[0]
    ndwi_delta = ndwi_delta[0]
    ndvi_epochs = ndvi_epochs[0]  # [T, H, W]
    ndwi_epochs = ndwi_epochs[0]

    def crop(t2d):
        if pad_h or pad_w:
            t2d = t2d[: t2d.shape[0] - pad_h or None, : t2d.shape[1] - pad_w or None]
        return t2d[: orig_hw[0], : orig_hw[1]]

    probability = crop(probability)
    ndvi_delta = crop(ndvi_delta)
    ndwi_delta = crop(ndwi_delta)
    ndvi_epochs = torch.stack([crop(ndvi_epochs[t]) for t in range(ndvi_epochs.shape[0])])
    ndwi_epochs = torch.stack([crop(ndwi_epochs[t]) for t in range(ndwi_epochs.shape[0])])

    return (
        probability.cpu().numpy(),
        ndvi_delta.cpu().numpy(),
        ndwi_delta.cpu().numpy(),
        ndvi_epochs.cpu().numpy(),
        ndwi_epochs.cpu().numpy(),
    )


def run_inference(checkpoint_path, image_earlier_path, image_later_path, device=None, threshold=0.5):
    """High-level entry point used by the API layer.

    image_earlier_path = older image (epoch B)
    image_later_path   = newer image (epoch A)

    Returns a dict of numpy arrays: probability, ndvi_delta, ndwi_delta,
    ndvi_epochs [2,H,W] (earlier, later), ndwi_epochs [2,H,W], plus orig_hw.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    x, qa, orig_hw = prepare_pair(image_later_path, image_earlier_path, device)

    probability, ndvi_delta, ndwi_delta, ndvi_epochs, ndwi_epochs = run_whole_image(
        model, x, qa, orig_hw
    )

    change_mask = probability >= threshold
    changed_fraction = float(change_mask.mean())

    return {
        "probability": probability,
        "change_mask": change_mask,
        "ndvi_delta": ndvi_delta,
        "ndwi_delta": ndwi_delta,
        "ndvi_earlier": ndvi_epochs[1],
        "ndvi_later": ndvi_epochs[0],
        "ndwi_earlier": ndwi_epochs[1],
        "ndwi_later": ndwi_epochs[0],
        "changed_fraction": changed_fraction,
        "orig_hw": orig_hw,
    }
