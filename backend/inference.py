"""
Inference layer for the SatelliteChangeNet deployment.

SUPPORTED MODES
---------------

1. RGB MODE
   User uploads:
       - one earlier JPG/PNG image
       - one later JPG/PNG image

   Uses:
       checkpoint_final.pth

   Expected model input:
       3 channels


2. MSI MODE
   User uploads ONE ZIP containing two Sentinel-2 acquisitions.

   Preferred structure:

       New folder/
       ├── earlier/
       │   ├── ..._B01.tif
       │   ├── ..._B02.tif
       │   ├── ..._B03.tif
       │   ├── ..._B04.tif
       │   ├── ..._B05.tif
       │   ├── ..._B06.tif
       │   ├── ..._B07.tif
       │   ├── ..._B08.tif
       │   ├── ..._B8A.tif
       │   ├── ..._B09.tif
       │   ├── ..._B10.tif
       │   ├── ..._B11.tif
       │   └── ..._B12.tif
       │
       └── later/
           ├── ..._B01.tif
           ├── ..._B02.tif
           ├── ...
           ├── ..._B12.tif
           └── ..._B8A.tif

   Also supports:
       imgs_1 / imgs_2
       imgs1 / imgs2
       T1 / T2
       date1 / date2
       before / after
       old / new

   Uses:
       checkpoint_final (1).pth

   Expected model input:
       13 Sentinel-2 MSI channels

IMPORTANT
---------
The model architecture is NOT changed here.

The checkpoint determines:
    - architecture configuration
    - number of input channels
    - trained weights

No synthetic MSI channels are created from RGB images.
"""

# ============================================================
# IMPORTS
# ============================================================

import json
import re
import shutil
import tempfile
import zipfile

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from model import SatelliteChangeNet


# ============================================================
# MODULE INFORMATION
# ============================================================

INFERENCE_FILE = Path(__file__).resolve()

INFERENCE_VERSION = "two-checkpoint-mode-v1"

# ============================================================
# MODEL CONFIGURATION OVERRIDES
# ============================================================

RGB_CONFIG_OVERRIDES = {
    "in_channels": 3,
    "red_index": 0,
    "green_index": 1,
    "nir_index": 3,
}

MSI_CONFIG_OVERRIDES = {
    "in_channels": 13,
    "red_index": 3,
    "green_index": 2,
    "nir_index": 7,
}


# ============================================================
# SENTINEL-2 BAND ORDER
# ============================================================
#
# IMPORTANT:
# This order must match the numerical sorting used during
# training.
#
# Correct order:
#
# B01
# B02
# B03
# B04
# B05
# B06
# B07
# B08
# B8A
# B09
# B10
# B11
# B12
#
# Total = 13 channels
#
# ============================================================

MSI_BANDS = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
)


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_stack(array: np.ndarray) -> np.ndarray:
    """
    Normalize an H,W,C array to approximately [0,1].

    Behavior:
        max <= 1       -> assume already normalized
        max <= 255     -> divide by 255
        otherwise      -> 99.5 percentile normalization
    """

    array = np.asarray(array).astype(
        np.float32,
        copy=False,
    )

    # Replace NaN / Inf
    array[~np.isfinite(array)] = 0.0

    maximum = float(
        array.max(initial=0.0)
    )

    if maximum <= 1.0:
        return np.clip(
            array,
            0.0,
            1.0,
        )

    if maximum <= 255.0:
        return np.clip(
            array / 255.0,
            0.0,
            1.0,
        )

    # Per-channel percentile normalization
    flat = array.reshape(
        -1,
        array.shape[-1],
    )

    scale = np.percentile(
        flat,
        99.5,
        axis=0,
    )

    scale = np.maximum(
        scale,
        1e-6,
    ).reshape(
        1,
        1,
        -1,
    )

    return np.clip(
        array / scale,
        0.0,
        1.0,
    )


# ============================================================
# GENERIC FILE READER
# ============================================================

def _read_single_file(path: Path) -> np.ndarray:
    """
    Read a raster/image file and return H,W,C float32 data.

    Supported:
        .npy
        .tif
        .tiff
        JPG / PNG / other PIL-readable images
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {path}"
        )

    suffix = path.suffix.lower()

    # --------------------------------------------------------
    # NumPy
    # --------------------------------------------------------

    if suffix == ".npy":
        array = np.load(path)

    # --------------------------------------------------------
    # GeoTIFF / TIFF
    # --------------------------------------------------------

    elif suffix in {".tif", ".tiff"}:

        try:
            import rasterio

            with rasterio.open(path) as src:
                array = src.read()

        except Exception:

            # Fallback to PIL
            with Image.open(path) as image:
                array = np.asarray(image)

    # --------------------------------------------------------
    # Normal image
    # --------------------------------------------------------

    else:

        with Image.open(path) as image:
            array = np.asarray(
                image.convert("RGB")
            )

    array = np.asarray(array).copy()

    # --------------------------------------------------------
    # H,W
    # --------------------------------------------------------

    if array.ndim == 2:
        array = array[..., None]

    # --------------------------------------------------------
    # Either:
    #
    # H,W,C
    #
    # or rasterio:
    #
    # C,H,W
    # --------------------------------------------------------

    elif array.ndim == 3:

        # Rasterio C,H,W detection.
        #
        # Sentinel bands have small channel count,
        # while spatial dimensions are normally larger.
        if (
            array.shape[0] <= 32
            and array.shape[1] > 32
            and array.shape[2] > 32
        ):
            array = np.transpose(
                array,
                (1, 2, 0),
            )

    else:
        raise ValueError(
            f"Unsupported image shape {array.shape} "
            f"in {path}"
        )

    return array.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# RGB READER
# ============================================================

def read_rgb(path: Path) -> np.ndarray:
    """
    Read a normal RGB image.

    Returns:
        H,W,3
    """

    array = _read_single_file(
        Path(path)
    )

    channels = array.shape[-1]

    if channels == 1:

        array = np.repeat(
            array,
            3,
            axis=-1,
        )

    elif channels >= 3:

        array = array[..., :3]

    else:
        raise ValueError(
            f"RGB image has invalid shape "
            f"{array.shape}"
        )

    return _normalize_stack(array)


# ============================================================
# SENTINEL BAND IDENTIFICATION
# ============================================================

def _band_from_name(path: Path):
    """
    Extract Sentinel-2 band name from filename.

    Correctly handles:
        B01 ... B12
        B8A

    B8A is checked before B08.
    """

    name = (
        path.name
        .upper()
        .replace("-", "_")
    )

    candidates = (
        "B8A",
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B09",
        "B10",
        "B11",
        "B12",
    )

    for band in candidates:

        pattern = (
            rf"(?<![A-Z0-9])"
            rf"{re.escape(band)}"
            rf"(?![A-Z0-9])"
        )

        if re.search(
            pattern,
            name,
        ):
            return band

    return None


# ============================================================
# DATE / TEMPORAL GROUP IDENTIFICATION
# ============================================================

def _date_group(path: Path):
    """
    Determine whether a file belongs to the earlier
    or later acquisition.

    IMPORTANT:
    Only exact path components are used for folder-based
    detection.

    This prevents:

        New folder/

    from accidentally being interpreted as:

        new = later

    Supported folder names:

        earlier
        later

        imgs_1
        imgs_2

        imgs1
        imgs2

        T1
        T2

        date1
        date2

        before
        after

        old
        new
    """

    path = Path(path)

    # --------------------------------------------------------
    # Exact path components
    # --------------------------------------------------------

    parts = {
        str(part).strip().lower()
        for part in path.parts
        if str(part).strip()
    }

    earlier_markers = {
        "earlier",
        "imgs_1",
        "imgs1",
        "t1",
        "date1",
        "before",
        "old",
    }

    later_markers = {
        "later",
        "imgs_2",
        "imgs2",
        "t2",
        "date2",
        "after",
        "new",
    }

    if parts.intersection(
        earlier_markers
    ):
        return "earlier"

    if parts.intersection(
        later_markers
    ):
        return "later"

    # --------------------------------------------------------
    # Filename fallback
    #
    # This is intentionally strict.
    # --------------------------------------------------------

    filename = path.name.lower()

    earlier_pattern = (
        r"(^|[_\-. ])"
        r"(earlier|before|old|t1|date1)"
        r"([_\-. ]|$)"
    )

    later_pattern = (
        r"(^|[_\-. ])"
        r"(later|after|t2|date2)"
        r"([_\-. ]|$)"
    )

    if re.search(
        earlier_pattern,
        filename,
    ):
        return "earlier"

    if re.search(
        later_pattern,
        filename,
    ):
        return "later"

    return None


# ============================================================
# READ COMPLETE 13-BAND DATE
# ============================================================

def _read_band_set(paths, label):
    """
    Read one complete Sentinel-2 acquisition.

    Requires exactly one file for each of:

        B01
        B02
        B03
        B04
        B05
        B06
        B07
        B08
        B8A
        B09
        B10
        B11
        B12
    """

    paths = [
        Path(p)
        for p in paths
    ]

    by_band = {}

    # --------------------------------------------------------
    # Identify bands
    # --------------------------------------------------------

    for path in paths:

        band = _band_from_name(path)

        if band is None:
            continue

        if band in by_band:
            raise ValueError(
                f"Duplicate {label} band "
                f"{band}:\n"
                f"{by_band[band]}\n"
                f"{path}"
            )

        by_band[band] = path

    # --------------------------------------------------------
    # Missing bands
    # --------------------------------------------------------

    missing = [
        band
        for band in MSI_BANDS
        if band not in by_band
    ]

    if missing:

        detected = sorted(
            by_band.keys()
        )

        raise ValueError(
            f"{label} is missing Sentinel-2 bands.\n"
            f"Missing: {', '.join(missing)}\n"
            f"Detected: {', '.join(detected)}"
        )

    # --------------------------------------------------------
    # Read every band
    # --------------------------------------------------------

    arrays = []

    target_h = 0
    target_w = 0

    for band in MSI_BANDS:

        path = by_band[band]

        arr = _read_single_file(path)

        if arr.ndim != 3:
            raise ValueError(
                f"Invalid raster shape for "
                f"{path.name}: {arr.shape}"
            )

        # A Sentinel-2 individual band file must be one channel.
        if arr.shape[-1] != 1:

            raise ValueError(
                f"{path.name} contains "
                f"{arr.shape[-1]} channels. "
                f"Expected exactly 1 channel "
                f"for Sentinel-2 band {band}."
            )

        arr = arr[..., 0]

        arrays.append(arr)

        target_h = max(
            target_h,
            arr.shape[0],
        )

        target_w = max(
            target_w,
            arr.shape[1],
        )

    # --------------------------------------------------------
    # Spatially align all bands within this date
    # --------------------------------------------------------

    aligned = []

    for arr in arrays:

        if arr.shape == (
            target_h,
            target_w,
        ):
            aligned.append(arr)
            continue

        tensor = torch.from_numpy(
            arr
        ).float()[None, None]

        tensor = F.interpolate(
            tensor,
            size=(
                target_h,
                target_w,
            ),
            mode="bilinear",
            align_corners=False,
        )

        aligned.append(
            tensor[0, 0].numpy()
        )

    # --------------------------------------------------------
    # H,W,13
    # --------------------------------------------------------

    stack = np.stack(
        aligned,
        axis=-1,
    )

    if stack.shape[-1] != 13:
        raise ValueError(
            f"{label} produced "
            f"{stack.shape[-1]} channels. "
            f"Expected 13."
        )

    return _normalize_stack(stack)


# ============================================================
# MSI ZIP PREPARATION
# ============================================================

def prepare_msi_zip(zip_path: Path):
    """
    Extract and parse a Sentinel-2 MSI ZIP.

    The ZIP can have a wrapper directory.

    Example:

        New folder.zip

        New folder/
        ├── earlier/
        │   ├── B01.tif
        │   ├── ...
        │   └── B8A.tif
        │
        └── later/
            ├── B01.tif
            ├── ...
            └── B8A.tif

    Returns:

        earlier_stack,
        later_stack,
        temporary_directory
    """

    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(
            f"MSI ZIP does not exist:\n"
            f"{zip_path}"
        )

    if zip_path.suffix.lower() != ".zip":
        raise ValueError(
            "MSI input must be a ZIP file."
        )

    work_dir = Path(
        tempfile.mkdtemp(
            prefix="oscd_msi_"
        )
    )

    try:

        # ====================================================
        # OPEN ZIP
        # ====================================================

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as zf:

            # ------------------------------------------------
            # Test ZIP integrity
            # ------------------------------------------------

            bad_file = zf.testzip()

            if bad_file is not None:
                raise ValueError(
                    f"MSI ZIP is corrupted. "
                    f"First bad file: {bad_file}"
                )

            members = zf.namelist()

            if not members:
                raise ValueError(
                    "MSI ZIP is empty."
                )

            # ------------------------------------------------
            # Security check
            # ------------------------------------------------

            bad_paths = []

            for member in members:

                member_path = Path(member)

                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                ):
                    bad_paths.append(
                        member
                    )

            if bad_paths:
                raise ValueError(
                    "ZIP contains unsafe paths."
                )

            # ------------------------------------------------
            # Extract
            # ------------------------------------------------

            zf.extractall(
                work_dir
            )

        # ====================================================
        # FIND TIFF FILES
        # ====================================================

        tiff_files = sorted(
            p
            for p in work_dir.rglob("*")
            if (
                p.is_file()
                and p.suffix.lower()
                in {".tif", ".tiff"}
            )
        )

        if not tiff_files:

            raise ValueError(
                "MSI ZIP contains no "
                "TIFF/TIF files."
            )

        # ====================================================
        # GROUP BY DATE
        # ====================================================

        groups = {
            "earlier": [],
            "later": [],
        }

        unclassified = []

        for path in tiff_files:

            relative_path = (
                path.relative_to(
                    work_dir
                )
            )

            group = _date_group(
                relative_path
            )

            if group is None:

                unclassified.append(
                    str(relative_path)
                )

            else:

                groups[group].append(
                    path
                )

        # ====================================================
        # IMPORTANT DIAGNOSTIC
        # ====================================================

        if (
            not groups["earlier"]
            or not groups["later"]
        ):

            earlier_names = [
                str(
                    p.relative_to(
                        work_dir
                    )
                )
                for p in groups["earlier"]
            ]

            later_names = [
                str(
                    p.relative_to(
                        work_dir
                    )
                )
                for p in groups["later"]
            ]

            all_names = [
                str(
                    p.relative_to(
                        work_dir
                    )
                )
                for p in tiff_files
            ]

            raise ValueError(
                "\n"
                "MSI ZIP DATE DETECTION FAILED\n"
                "================================\n"
                f"ZIP: {zip_path.name}\n"
                f"TIFF files found: {len(tiff_files)}\n"
                f"Earlier files: {len(groups['earlier'])}\n"
                f"Later files: {len(groups['later'])}\n"
                "\n"
                "Expected a structure such as:\n"
                "  earlier/*.tif\n"
                "  later/*.tif\n"
                "\n"
                "or:\n"
                "  imgs_1/*.tif\n"
                "  imgs_2/*.tif\n"
                "\n"
                "Detected TIFF files:\n"
                + "\n".join(all_names)
            )

        # ====================================================
        # BAND COUNT CHECK
        # ====================================================

        if len(groups["earlier"]) != 13:

            raise ValueError(
                "Earlier acquisition must contain "
                f"13 TIFF band files, but found "
                f"{len(groups['earlier'])}.\n\n"
                "Earlier files:\n"
                + "\n".join(
                    str(
                        p.relative_to(
                            work_dir
                        )
                    )
                    for p in groups["earlier"]
                )
            )

        if len(groups["later"]) != 13:

            raise ValueError(
                "Later acquisition must contain "
                f"13 TIFF band files, but found "
                f"{len(groups['later'])}.\n\n"
                "Later files:\n"
                + "\n".join(
                    str(
                        p.relative_to(
                            work_dir
                        )
                    )
                    for p in groups["later"]
                )
            )

        # ====================================================
        # READ BANDS
        # ====================================================

        earlier = _read_band_set(
            groups["earlier"],
            "earlier/T1",
        )

        later = _read_band_set(
            groups["later"],
            "later/T2",
        )

        # ====================================================
        # FINAL VALIDATION
        # ====================================================

        if earlier.ndim != 3:
            raise ValueError(
                f"Earlier MSI stack has invalid "
                f"shape: {earlier.shape}"
            )

        if later.ndim != 3:
            raise ValueError(
                f"Later MSI stack has invalid "
                f"shape: {later.shape}"
            )

        if earlier.shape[-1] != 13:
            raise ValueError(
                f"Earlier MSI stack has "
                f"{earlier.shape[-1]} channels. "
                f"Expected 13."
            )

        if later.shape[-1] != 13:
            raise ValueError(
                f"Later MSI stack has "
                f"{later.shape[-1]} channels. "
                f"Expected 13."
            )

        return (
            earlier,
            later,
            work_dir,
        )

    except Exception:

        shutil.rmtree(
            work_dir,
            ignore_errors=True,
        )

        raise


# ============================================================
# QUALITY-ASSURANCE PROXY
# ============================================================

def qa_proxy(
    image: torch.Tensor,
) -> torch.Tensor:
    """
    Generate a simple QA validity mask.

    For MSI:
        B02 = channel 1
        B03 = channel 2
        B04 = channel 3

    For RGB:
        first three channels are used.
    """

    channels = image.shape[0]

    # --------------------------------------------------------
    # MSI
    # --------------------------------------------------------

    if channels >= 13:

        rgb = image[
            [1, 2, 3]
        ]

    # --------------------------------------------------------
    # RGB / other
    # --------------------------------------------------------

    else:

        rgb = image[
            :min(3, channels)
        ]

    # --------------------------------------------------------
    # Single channel
    # --------------------------------------------------------

    if rgb.shape[0] == 1:

        luminance = rgb[0]

        saturation = torch.zeros_like(
            luminance
        )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    else:

        luminance = (
            0.299 * rgb[0]
            + 0.587 * rgb[
                min(
                    1,
                    rgb.shape[0] - 1,
                )
            ]
            + 0.114 * rgb[
                min(
                    2,
                    rgb.shape[0] - 1,
                )
            ]
        )

        saturation = (
            rgb.max(dim=0).values
            - rgb.min(dim=0).values
        )

    # --------------------------------------------------------
    # Simple cloud/shadow proxy
    # --------------------------------------------------------

    possible_cloud = (
        (luminance > 0.97)
        & (saturation < 0.12)
    )

    possible_shadow = (
        (luminance < 0.035)
        & (saturation < 0.20)
    )

    invalid = (
        possible_cloud
        | possible_shadow
    ).float().unsqueeze(0)

    invalid = F.max_pool2d(
        invalid.unsqueeze(0),
        kernel_size=3,
        stride=1,
        padding=1,
    )[0]

    return 1.0 - invalid


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(
    checkpoint_path,
    device,
    mode,
):
    """
    Load SatelliteChangeNet and its checkpoint configuration.
    
    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the checkpoint file
    device : torch.device
        Device to load model on
    mode : str
        'rgb' or 'msi' - determines configuration overrides
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Checkpoint with embedded configuration
    # --------------------------------------------------------

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

        model_config = checkpoint.get(
            "model_config"
        )

    # --------------------------------------------------------
    # Raw state dict
    # --------------------------------------------------------

    else:

        state_dict = checkpoint

        config_path = (
            checkpoint_path.parent
            / "config.json"
        )

        if not config_path.exists():

            raise FileNotFoundError(
                f"{checkpoint_path} does not contain "
                f"an embedded model_config and "
                f"no config.json was found next to it."
            )

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as f:

            config_data = json.load(f)

        model_config = config_data.get(
            "model_config"
        )

    model_config = (
        model_config
        or {}
    )

    # --------------------------------------------------------
    # Apply mode-specific configuration overrides
    # --------------------------------------------------------

    mode = str(mode).lower().strip()

    if mode == "rgb":
        model_config.update(RGB_CONFIG_OVERRIDES)
    elif mode == "msi":
        model_config.update(MSI_CONFIG_OVERRIDES)
    else:
        raise ValueError(
            f"Unsupported mode: {mode}"
        )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = SatelliteChangeNet(
        **model_config
    ).to(device)

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return (
        model,
        model_config,
    )


# ============================================================
# PREPARE TWO IMAGES
# ============================================================

def prepare_pair_arrays(
    a,
    b,
    device,
):
    """
    Convert two H,W,C arrays into model input.

    a = earlier
    b = later
    """

    if a.ndim != 3:
        raise ValueError(
            "Earlier input must be "
            "an H,W,C array."
        )

    if b.ndim != 3:
        raise ValueError(
            "Later input must be "
            "an H,W,C array."
        )

    if a.shape[-1] != b.shape[-1]:
        raise ValueError(
            "Channel mismatch: "
            f"earlier={a.shape[-1]}, "
            f"later={b.shape[-1]}"
        )

    # --------------------------------------------------------
    # H,W,C -> C,H,W
    # --------------------------------------------------------

    a_tensor = torch.from_numpy(
        np.ascontiguousarray(a)
    ).float().permute(
        2,
        0,
        1,
    )

    b_tensor = torch.from_numpy(
        np.ascontiguousarray(b)
    ).float().permute(
        2,
        0,
        1,
    )

    # --------------------------------------------------------
    # Align later image to earlier image dimensions
    # --------------------------------------------------------

    if (
        b_tensor.shape[-2:]
        != a_tensor.shape[-2:]
    ):

        b_tensor = F.interpolate(
            b_tensor.unsqueeze(0),
            size=a_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0]

    # --------------------------------------------------------
    # QA
    # --------------------------------------------------------

    qa_a = qa_proxy(
        a_tensor
    )

    qa_b = qa_proxy(
        b_tensor
    )

    # --------------------------------------------------------
    # Model expects temporal stack
    #
    # Current convention:
    #     T2 = later
    #     T1 = earlier
    #
    # --------------------------------------------------------

    x = torch.stack(
        (
            b_tensor,
            a_tensor,
        ),
        dim=0,
    ).unsqueeze(0).to(
        device
    )

    qa = torch.stack(
        (
            qa_b,
            qa_a,
        ),
        dim=0,
    ).unsqueeze(0).to(
        device
    )

    return (
        x,
        qa,
        a_tensor.shape[-2:],
    )


# ============================================================
# PREPARE INDIVIDUAL FILE PAIR
# ============================================================

def prepare_pair_files(
    earlier_path,
    later_path,
    mode,
    device,
):
    """
    Prepare two standalone input files.

    RGB:
        JPG/PNG/etc.

    MSI:
        each file must already be a 13-channel
        raster/array.
    """

    mode = str(
        mode
    ).lower().strip()

    # ========================================================
    # RGB
    # ========================================================

    if mode == "rgb":

        earlier = read_rgb(
            Path(earlier_path)
        )

        later = read_rgb(
            Path(later_path)
        )

    # ========================================================
    # MSI
    # ========================================================

    elif mode == "msi":

        earlier = _read_single_file(
            Path(earlier_path)
        )

        later = _read_single_file(
            Path(later_path)
        )

        if earlier.shape[-1] != 13:

            raise ValueError(
                "Internal MSI earlier input must "
                f"contain exactly 13 channels, "
                f"but got {earlier.shape[-1]}."
            )

        if later.shape[-1] != 13:

            raise ValueError(
                "Internal MSI later input must "
                f"contain exactly 13 channels, "
                f"but got {later.shape[-1]}."
            )

        earlier = _normalize_stack(
            earlier
        )

        later = _normalize_stack(
            later
        )

    else:

        raise ValueError(
            f"Unsupported mode: {mode}"
        )

    return prepare_pair_arrays(
        earlier,
        later,
        device,
    )


# ============================================================
# PAD INPUT TO MULTIPLE
# ============================================================

def pad_to_multiple(
    x,
    qa,
    multiple=32,
):
    """
    Pad spatial dimensions to a multiple of 32.
    """

    height, width = (
        x.shape[-2:]
    )

    pad_h = (
        multiple
        - height % multiple
    ) % multiple

    pad_w = (
        multiple
        - width % multiple
    ) % multiple

    if pad_h or pad_w:

        x = F.pad(
            x,
            (
                0,
                pad_w,
                0,
                pad_h,
            ),
            value=0.0,
        )

        qa = F.pad(
            qa,
            (
                0,
                pad_w,
                0,
                pad_h,
            ),
            value=0.0,
        )

    return (
        x,
        qa,
        pad_h,
        pad_w,
    )


# ============================================================
# WHOLE IMAGE INFERENCE
# ============================================================

@torch.no_grad()
def run_whole_image(
    model,
    x,
    qa,
    orig_hw,
    multiple=32,
):
    """
    Run model inference on the complete image.
    """

    x_padded, qa_padded, pad_h, pad_w = (
        pad_to_multiple(
            x,
            qa,
            multiple,
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    (
        logits,
        ndvi_delta,
        ndwi_delta,
        ndvi_epochs,
        ndwi_epochs,
    ) = model(
        x_padded,
        qa_padded,
        return_indices=True,
    )

    # --------------------------------------------------------
    # Change probability
    # --------------------------------------------------------

    probability = torch.sigmoid(
        logits
    )[0, 0]

    # --------------------------------------------------------
    # Auxiliary outputs
    # --------------------------------------------------------

    ndvi_delta = ndvi_delta[0]
    ndwi_delta = ndwi_delta[0]

    ndvi_epochs = ndvi_epochs[0]
    ndwi_epochs = ndwi_epochs[0]

    # --------------------------------------------------------
    # Crop padding
    # --------------------------------------------------------

    def crop(tensor):

        if pad_h or pad_w:

            h_end = (
                tensor.shape[0]
                - pad_h
                if pad_h
                else None
            )

            w_end = (
                tensor.shape[1]
                - pad_w
                if pad_w
                else None
            )

            tensor = tensor[
                :h_end,
                :w_end,
            ]

        return tensor[
            :orig_hw[0],
            :orig_hw[1],
        ]

    probability = crop(
        probability
    )

    ndvi_delta = crop(
        ndvi_delta
    )

    ndwi_delta = crop(
        ndwi_delta
    )

    # --------------------------------------------------------
    # NDVI epochs
    # --------------------------------------------------------

    ndvi_epochs = torch.stack(
        [
            crop(
                ndvi_epochs[t]
            )
            for t in range(
                ndvi_epochs.shape[0]
            )
        ]
    )

    # --------------------------------------------------------
    # NDWI epochs
    # --------------------------------------------------------

    ndwi_epochs = torch.stack(
        [
            crop(
                ndwi_epochs[t]
            )
            for t in range(
                ndwi_epochs.shape[0]
            )
        ]
    )

    # --------------------------------------------------------
    # QA output
    # --------------------------------------------------------

    qa_orig = F.interpolate(
        qa_padded.reshape(
            -1,
            1,
            *qa_padded.shape[-2:],
        ),
        size=x.shape[-2:],
        mode="nearest",
    ).reshape(
        qa_padded.shape[0],
        qa_padded.shape[1],
        *x.shape[-2:],
    )[0]

    qa_orig = torch.stack(
        [
            crop(
                qa_orig[t]
            )
            for t in range(
                qa_orig.shape[0]
            )
        ]
    )

    return (
        probability.cpu().numpy(),
        ndvi_delta.cpu().numpy(),
        ndwi_delta.cpu().numpy(),
        ndvi_epochs.cpu().numpy(),
        ndwi_epochs.cpu().numpy(),
        qa_orig.cpu().numpy(),
    )


# ============================================================
# STANDARD RGB / FILE INFERENCE
# ============================================================

def run_inference(
    checkpoint_path,
    image_earlier_path,
    image_later_path,
    mode="rgb",
    device=None,
    threshold=0.5,
):
    """
    Standard inference entry point.

    Used primarily for RGB mode.

    mode='rgb':
        3-channel checkpoint

    mode='msi':
        13-channel checkpoint
    """

    mode = str(
        mode
    ).lower().strip()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = device or torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    model, model_config = load_model(
        checkpoint_path,
        device,
        mode,
    )

    expected_channels = int(
        model_config.get(
            "in_channels",
            3,
        )
    )

    required_channels = (
        3
        if mode == "rgb"
        else 13
    )

    if expected_channels != required_channels:

        raise ValueError(
            f"Checkpoint "
            f"{Path(checkpoint_path).name} "
            f"expects {expected_channels} "
            f"channels, but mode='{mode}' "
            f"supplies {required_channels} "
            f"channels."
        )

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    (
        x,
        qa,
        orig_hw,
    ) = prepare_pair_files(
        image_earlier_path,
        image_later_path,
        mode,
        device,
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    (
        probability,
        ndvi_delta,
        ndwi_delta,
        ndvi_epochs,
        ndwi_epochs,
        qa_orig,
    ) = run_whole_image(
        model,
        x,
        qa,
        orig_hw,
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    change_mask = (
        probability >= threshold
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "probability": probability,
        "change_mask": change_mask,

        "ndvi_delta": ndvi_delta,
        "ndwi_delta": ndwi_delta,

        "ndvi_earlier": ndvi_epochs[1],
        "ndvi_later": ndvi_epochs[0],

        "ndwi_earlier": ndwi_epochs[1],
        "ndwi_later": ndwi_epochs[0],

        "qa_earlier": qa_orig[1],
        "qa_later": qa_orig[0],

        "changed_fraction": float(
            change_mask.mean()
        ),

        "orig_hw": orig_hw,

        "mode": mode,

        "checkpoint": Path(
            checkpoint_path
        ).name,

        "inference_version":
            INFERENCE_VERSION,
    }


# ============================================================
# MSI ZIP INFERENCE
# ============================================================

def run_msi_zip_inference(
    checkpoint_path,
    zip_path,
    device=None,
    threshold=0.5,
):
    """
    Run inference from ONE MSI ZIP.

    The ZIP must contain:

        earlier = 13 Sentinel-2 bands
        later   = 13 Sentinel-2 bands
    """

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = device or torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    model, model_config = load_model(
        checkpoint_path,
        device,
        "msi",
    )

    expected_channels = int(
        model_config.get(
            "in_channels",
            13,
        )
    )

    if expected_channels != 13:

        raise ValueError(
            "MSI checkpoint must expect "
            f"13 channels, but this checkpoint "
            f"expects {expected_channels}."
        )

    # --------------------------------------------------------
    # Parse ZIP
    # --------------------------------------------------------

    earlier, later, work_dir = (
        prepare_msi_zip(
            Path(zip_path)
        )
    )

    try:

        # ----------------------------------------------------
        # Final shape validation
        # ----------------------------------------------------

        if earlier.shape[-1] != 13:

            raise ValueError(
                "Earlier MSI acquisition contains "
                f"{earlier.shape[-1]} channels. "
                "Expected 13."
            )

        if later.shape[-1] != 13:

            raise ValueError(
                "Later MSI acquisition contains "
                f"{later.shape[-1]} channels. "
                "Expected 13."
            )

        # ----------------------------------------------------
        # Prepare model input
        # ----------------------------------------------------

        (
            x,
            qa,
            orig_hw,
        ) = prepare_pair_arrays(
            earlier,
            later,
            device,
        )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        (
            probability,
            ndvi_delta,
            ndwi_delta,
            ndvi_epochs,
            ndwi_epochs,
            qa_orig,
        ) = run_whole_image(
            model,
            x,
            qa,
            orig_hw,
        )

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        change_mask = (
            probability >= threshold
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        return {
            "probability": probability,
            "change_mask": change_mask,

            "ndvi_delta": ndvi_delta,
            "ndwi_delta": ndwi_delta,

            "ndvi_earlier": ndvi_epochs[1],
            "ndvi_later": ndvi_epochs[0],

            "ndwi_earlier": ndwi_epochs[1],
            "ndwi_later": ndwi_epochs[0],

            "qa_earlier": qa_orig[1],
            "qa_later": qa_orig[0],

            "changed_fraction": float(
                change_mask.mean()
            ),

            "orig_hw": orig_hw,

            "mode": "msi",

            "checkpoint": Path(
                checkpoint_path
            ).name,

            "inference_version":
                INFERENCE_VERSION,
        }

    finally:

        # ----------------------------------------------------
        # Always remove temporary extraction directory
        # ----------------------------------------------------

        shutil.rmtree(
            work_dir,
            ignore_errors=True,
        )


# ============================================================
# RGB PREVIEW GENERATION
# ============================================================

def get_aligned_rgb_pair(
    earlier_path,
    later_path,
    mode="rgb",
):
    """
    Return aligned RGB preview arrays.

    RGB mode:
        returns the original RGB images.

    MSI mode:
        converts true Sentinel-2 bands to:
            RGB = B04, B03, B02
    """

    mode = str(
        mode
    ).lower().strip()

    # ========================================================
    # RGB
    # ========================================================

    if mode == "rgb":

        earlier = read_rgb(
            Path(earlier_path)
        )

        later = read_rgb(
            Path(later_path)
        )

        # Align earlier to later
        if (
            earlier.shape[:2]
            != later.shape[:2]
        ):

            tensor = torch.from_numpy(
                np.ascontiguousarray(
                    earlier
                )
            ).float().permute(
                2,
                0,
                1,
            )

            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=later.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0]

            earlier = (
                tensor
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

        return (
            later,
            earlier,
        )

    # ========================================================
    # MSI
    # ========================================================

    if mode == "msi":

        earlier = _normalize_stack(
            _read_single_file(
                Path(earlier_path)
            )
        )

        later = _normalize_stack(
            _read_single_file(
                Path(later_path)
            )
        )

        if earlier.shape[-1] != 13:
            raise ValueError(
                "MSI earlier preview requires "
                "13 channels."
            )

        if later.shape[-1] != 13:
            raise ValueError(
                "MSI later preview requires "
                "13 channels."
            )

        # ----------------------------------------------------
        # Align earlier to later
        # ----------------------------------------------------

        if (
            earlier.shape[:2]
            != later.shape[:2]
        ):

            tensor = torch.from_numpy(
                np.ascontiguousarray(
                    earlier
                )
            ).float().permute(
                2,
                0,
                1,
            )

            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=later.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0]

            earlier = (
                tensor
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

        # ----------------------------------------------------
        # Sentinel-2 true colour:
        #
        # B04 = red   = index 3
        # B03 = green = index 2
        # B02 = blue  = index 1
        # ----------------------------------------------------

        later_rgb = later[
            ...,
            [3, 2, 1],
        ]

        earlier_rgb = earlier[
            ...,
            [3, 2, 1],
        ]

        return (
            later_rgb,
            earlier_rgb,
        )

    raise ValueError(
        f"Unsupported preview mode: {mode}"
    )


# ============================================================
# OPTIONAL DIRECT ZIP INSPECTION
# ============================================================

def inspect_msi_zip(
    zip_path: Path,
):
    """
    Inspect an MSI ZIP without loading the model.

    Useful for debugging uploads.

    Returns a dictionary describing:
        - ZIP filename
        - TIFF count
        - earlier count
        - later count
        - detected bands
        - missing bands
    """

    zip_path = Path(
        zip_path
    )

    if not zip_path.exists():
        raise FileNotFoundError(
            f"ZIP not found: {zip_path}"
        )

    work_dir = Path(
        tempfile.mkdtemp(
            prefix="oscd_msi_inspect_"
        )
    )

    try:

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as zf:

            bad_file = zf.testzip()

            if bad_file:
                raise ValueError(
                    f"Corrupt ZIP member: "
                    f"{bad_file}"
                )

            zf.extractall(
                work_dir
            )

        tiff_files = sorted(
            p
            for p in work_dir.rglob("*")
            if (
                p.is_file()
                and p.suffix.lower()
                in {".tif", ".tiff"}
            )
        )

        groups = {
            "earlier": [],
            "later": [],
        }

        unclassified = []

        for path in tiff_files:

            relative = path.relative_to(
                work_dir
            )

            group = _date_group(
                relative
            )

            if group:
                groups[group].append(
                    relative
                )
            else:
                unclassified.append(
                    relative
                )

        def band_info(paths):

            bands = {}

            for path in paths:

                band = _band_from_name(
                    path
                )

                if band:
                    bands.setdefault(
                        band,
                        []
                    ).append(
                        str(path)
                    )

            return bands

        earlier_bands = band_info(
            groups["earlier"]
        )

        later_bands = band_info(
            groups["later"]
        )

        earlier_missing = [
            band
            for band in MSI_BANDS
            if band
            not in earlier_bands
        ]

        later_missing = [
            band
            for band in MSI_BANDS
            if band
            not in later_bands
        ]

        return {
            "zip": str(zip_path),

            "inference_version":
                INFERENCE_VERSION,

            "tiff_count":
                len(tiff_files),

            "earlier_count":
                len(groups["earlier"]),

            "later_count":
                len(groups["later"]),

            "unclassified_count":
                len(unclassified),

            "earlier_bands":
                sorted(
                    earlier_bands.keys()
                ),

            "later_bands":
                sorted(
                    later_bands.keys()
                ),

            "earlier_missing":
                earlier_missing,

            "later_missing":
                later_missing,

            "valid":
                (
                    len(groups["earlier"]) == 13
                    and len(groups["later"]) == 13
                    and not earlier_missing
                    and not later_missing
                ),
        }

    finally:

        shutil.rmtree(
            work_dir,
            ignore_errors=True,
        )