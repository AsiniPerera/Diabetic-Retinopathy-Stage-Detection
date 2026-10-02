"""Inference helpers exported from the coursework notebook; no demonstration cells execute."""
import warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import cv2
from PIL import Image, ImageOps
from dr_preprocessing import PreprocessConfig
PIPELINE_VERSION = "1.0.1"

def read_rgb(
    path: str | Path,
    config: PreprocessConfig,
) -> tuple[np.ndarray, dict]:

    # Convert input to Path
    path = Path(path)

    # Check that the file exists
    if not path.is_file():
        raise ValueError(f"File does not exist: {path}")

    # Read file size
    file_bytes = path.stat().st_size

    # Reject empty files
    if file_bytes == 0:
        raise ValueError("Image file is empty.")

    # Reject very large files
    if file_bytes > config.max_file_mb * 1024**2:
        raise ValueError(f"File exceeds the {config.max_file_mb} MB limit.")

    try:

        # Convert Pillow safety warnings into errors
        with warnings.catch_warnings():

            warnings.simplefilter(
                "error", Image.DecompressionBombWarning
            )

            # Open the image
            with Image.open(path) as image:

                # Record original image information
                detected_format = image.format
                original_mode = image.mode
                original_width = image.width
                original_height = image.height

                # Accept JPEG and PNG only
                if detected_format not in {
                    "JPEG",
                    "PNG",
                }:
                    raise ValueError("Only JPEG and PNG images are supported.")

                # Reject images that are too small
                if min(image.size) < config.min_side:
                    raise ValueError("Image dimensions are too small.")

                # Reject extremely large decoded images
                if image.width * image.height > config.max_pixels:
                    raise ValueError("Image exceeds the maximum pixel limit.")

                # Reject multi-frame images
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Multi-frame images are unsupported.")

                # Fully decode the image
                image.load()

                # Correct EXIF orientation
                oriented = ImageOps.exif_transpose(image)

                # Detect transparency
                has_transparency = (
                    "A" in oriented.getbands()
                    or "transparency" in oriented.info
                )

                # Replace transparent regions with black
                if has_transparency:

                    rgba = oriented.convert("RGBA")

                    background = Image.new(
                        "RGBA", rgba.size, (0, 0, 0, 255)
                    )

                    oriented = Image.alpha_composite(background, rgba)

                # Convert accepted image to RGB uint8
                rgb = np.asarray(oriented.convert("RGB"), dtype=np.uint8)

    except (
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as error:

        raise ValueError(
            "Image dimensions were rejected by Pillow safety protection."
        ) from error

    except (
        OSError,
        SyntaxError,
        EOFError,
    ) as error:

        raise ValueError(f"Cannot decode '{path.name}': {error}") from error

    # Confirm that image contains 3 channels
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB conversion failed.")

    # Confirm uint8 representation
    if rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB image.")

    # Record validation metadata
    metadata = {
        "filename": path.name,
        "format": detected_format,
        "original_mode": original_mode,
        "original_width": int(original_width),
        "original_height": int(original_height),
        "validated_width": int(rgb.shape[1]),
        "validated_height": int(rgb.shape[0]),
        "channels": int(rgb.shape[2]),
        "dtype": str(rgb.dtype),
        "file_bytes": int(file_bytes),
        "file_size_mb": round(file_bytes / (1024**2), 3),
        "transparency_composited": bool(has_transparency),
        "orientation_policy": "EXIF transpose",
        "status": "valid",
        "pipeline_version": PIPELINE_VERSION,
    }

    # Return validated image and metadata
    return (np.ascontiguousarray(rgb), metadata)

def extract_retina(
    image: np.ndarray,
    config: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Extracts the visible retinal disc, builds a binary mask, and crops the bounding box."""
    # read image dimensions for mask coverage and crop boundaries
    h, w, c = image.shape

    # Convert to grayscale for initial retinal boundary detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold to separate light retinal pixels from dark outer background
    _, binary_mask = cv2.threshold(
        gray, config.mask_threshold, 255, cv2.THRESH_BINARY
    )

    # Morphological closing to fill internal dark lesions, vessels, and optic disc
    kernel_size = max(3, min(h, w) // 100)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Keep only the largest connected component (main retinal circle)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask
    )
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        full_mask = np.where(
            labels == largest_label, np.uint8(255), np.uint8(0)
        )
    else:
        full_mask = binary_mask

    # Calculate mask coverage fraction
    mask_fraction = float(np.count_nonzero(full_mask) / (h * w))

    # Trigger fallback if boundary detection fails or mask is too small
    if mask_fraction < config.mask_min_fraction:
        # retain the full image when a reliable retinal boundary is unavailable
        mask_fallback = True
        x0, y0, x1, y1 = 0, 0, w, h
        full_mask = np.ones((h, w), dtype=np.uint8) * 255
    else:
        mask_fallback = False
        y_indices, x_indices = np.where(full_mask > 0)

        # expand the detected mask bounds by the configured crop margin
        x0 = max(0, int(np.min(x_indices)) - config.crop_margin)
        y0 = max(0, int(np.min(y_indices)) - config.crop_margin)
        x1 = min(w, int(np.max(x_indices)) + 1 + config.crop_margin)
        y1 = min(h, int(np.max(y_indices)) + 1 + config.crop_margin)

    # Crop image and mask to bounding box
    cropped_rgb = image[y0:y1, x0:x1]
    cropped_mask = full_mask[y0:y1, x0:x1]

    # record the crop boundaries and whether fallback was required
    extraction_meta = {
        "original_width": int(w),
        "original_height": int(h),
        "cropped_width": int(cropped_rgb.shape[1]),
        "cropped_height": int(cropped_rgb.shape[0]),
        "mask_fraction": float(mask_fraction),
        "mask_fallback": bool(mask_fallback),
        "bbox_x0": int(x0),
        "bbox_y0": int(y0),
        "bbox_x1": int(x1),
        "bbox_y1": int(y1),
    }

    # return the crop, cropped mask, full-size mask, and measurements
    return cropped_rgb, cropped_mask, full_mask, extraction_meta

def resize_pad(
    image: np.ndarray, mask: np.ndarray, target_size: int = 224
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Resizes image and mask preserving aspect ratio, then pads symmetrically to target_size x target_size."""
    # read the height and width of the extracted retinal image
    h, w = image.shape[:2]

    # Calculate scale factor to match longest dimension to target_size
    scale_factor = target_size / float(max(h, w))
    new_w = max(1, int(round(w * scale_factor)))
    new_h = max(1, int(round(h * scale_factor)))

    # Use INTER_AREA for downscaling images (preserves detail without anti-aliasing artifacts)
    img_interp = (
        cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC
    )
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=img_interp)

    # Use INTER_NEAREST for binary masks to preserve exact 0/255 label boundaries
    mask_interp = cv2.INTER_NEAREST
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=mask_interp)

    # Calculate symmetric black padding
    pad_h = target_size - new_h
    pad_w = target_size - new_w

    # divide any odd padding pixel between opposite sides
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Pad image and mask with constant zero (black) borders
    prepared_image = cv2.copyMakeBorder(
        resized_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    prepared_mask = cv2.copyMakeBorder(
        resized_mask,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    # record dimensions, scaling, padding, and interpolation for reproducibility
    geometry_meta = {
        "original_height": int(h),
        "original_width": int(w),
        "resized_height": int(new_h),
        "resized_width": int(new_w),
        "target_size": int(target_size),
        "scale_factor": float(scale_factor),
        "padding_top": int(pad_top),
        "padding_bottom": int(pad_bottom),
        "padding_left": int(pad_left),
        "padding_right": int(pad_right),
        "image_interpolation": "INTER_AREA"
        if scale_factor < 1.0
        else "INTER_CUBIC",
        "mask_interpolation": "INTER_NEAREST",
    }

    # return the aligned model input, mask, and geometry record
    return prepared_image, prepared_mask, geometry_meta

def safe_uint8(values):
    return np.clip(values, 0, 255).astype(np.uint8)

def reconstruct_rgb(
    rgb: np.ndarray,
    mask: np.ndarray,
    config: PreprocessConfig,
) -> np.ndarray:
    """Clip and mask the classifier image without modifying the quality branch."""
    # Ensure pixel intensities are safely clipped to standard uint8 range [0, 255]
    output = safe_uint8(rgb)

    # Cast mask to boolean to zero out non-retinal outer background pixels
    mask_bool = mask.astype(bool)
    output[~mask_bool] = 0

    # Retrieve expected target dimension from configuration instance
    target_size = getattr(config, "size", output.shape[0])

    # Enforce spatial dimension integrity
    if output.shape != (target_size, target_size, 3):
        raise ValueError("Unexpected reconstructed image dimensions.")
    if mask.shape != (target_size, target_size):
        raise ValueError("Unexpected retinal-mask dimensions.")

    # Convert to contiguous array layout in memory for downstream PyTorch tensor conversion
    return np.ascontiguousarray(output)

@dataclass(frozen=True)
class QualityGateConfig:
    analysis_size: int = 224
    minimum_dimension: int = 128
    foreground_threshold: int = 10

    reject_coverage: float = 0.05
    review_coverage: float = 0.20

    reject_brightness_low: float = 15.0
    reject_brightness_high: float = 240.0
    review_brightness_low: float = 35.0
    review_brightness_high: float = 200.0

    reject_sharpness: float = 5.0
    review_sharpness: float = 50.0
    review_contrast: float = 15.0

    glare_value: int = 230
    glare_saturation: int = 40
    reject_glare_fraction: float = 0.15
    review_glare_fraction: float = 0.05


QUALITY_GATE_CONFIG = QualityGateConfig()


def quality_gate_result(status, reasons, metrics):
    # Rejected records contain no grade, confidence, or heatmap
    return {
        "quality_status": status,
        "classification_permitted": status != "rejected",
        "human_review_required": status == "review",
        "reasons": reasons,
        "metrics": metrics,
        "method": "Unvalidated image-quality heuristics",
        "retinal_identity_verified": False,
    }


def load_quality_rgb(image_path):
    # Validate and decode the file before quality assessment
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError("The uploaded image file was not found.")

    with Image.open(image_path) as source:
        # Correct orientation before inspecting dimensions
        oriented = ImageOps.exif_transpose(source)

        # Composite transparent pixels onto a black background
        if oriented.mode in ("RGBA", "LA") or "transparency" in oriented.info:
            rgba = oriented.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
            rgb = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            rgb = oriented.convert("RGB")

        return np.array(rgb, dtype=np.uint8, copy=True)


def assess_upload_quality(image_path, config=QUALITY_GATE_CONFIG):
    # Inspect the unenhanced image before model preprocessing
    try:
        rgb = load_quality_rgb(image_path)
    except (OSError, ValueError, Image.DecompressionBombError):
        return quality_gate_result(
            "rejected",
            ["The file could not be read as a supported image."],
            {},
        )

    height, width = rgb.shape[:2]
    metrics = {
        "original_width": int(width),
        "original_height": int(height),
    }

    if min(height, width) < config.minimum_dimension:
        return quality_gate_result(
            "rejected",
            ["Image resolution is below the configured minimum."],
            metrics,
        )

    # Resize proportionally so sharpness is measured at a consistent scale
    scale = config.analysis_size / max(height, width)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))

    analysis_rgb = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(analysis_rgb, cv2.COLOR_RGB2GRAY)

    # Estimate the largest visible foreground region
    foreground = (
        gray > config.foreground_threshold
    ).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        foreground,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return quality_gate_result(
            "rejected",
            ["No sufficiently visible foreground region was detected."],
            metrics,
        )

    largest_contour = max(contours, key=cv2.contourArea)
    region_mask = np.zeros_like(gray, dtype=np.uint8)

    # Fill the region so dark internal pixels remain part of the assessment
    cv2.drawContours(
        region_mask,
        [largest_contour],
        -1,
        color=255,
        thickness=cv2.FILLED,
    )

    coverage = float(np.mean(region_mask > 0))
    metrics["foreground_coverage"] = coverage

    if coverage < config.reject_coverage:
        return quality_gate_result(
            "rejected",
            ["Too little visible foreground area was detected."],
            metrics,
        )

    # Exclude the outer boundary from sharpness and intensity measurements
    interior = cv2.erode(
        region_mask,
        np.ones((5, 5), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0

    if np.count_nonzero(interior) < 32:
        return quality_gate_result(
            "rejected",
            ["Insufficient interior image area for quality assessment."],
            metrics,
        )

    values = gray[interior].astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    hsv = cv2.cvtColor(analysis_rgb, cv2.COLOR_RGB2HSV)

    # Bright, low-saturation pixels are glare candidates, not confirmed glare
    glare_candidates = (
        (hsv[..., 2] >= config.glare_value)
        & (hsv[..., 1] <= config.glare_saturation)
        & interior
    )

    metrics.update({
        "mean_brightness": float(values.mean()),
        "contrast_std": float(values.std()),
        "laplacian_variance": float(laplacian[interior].var()),
        "dark_fraction": float(np.mean(values < 15)),
        "bright_fraction": float(np.mean(values > 245)),
        "glare_candidate_fraction": float(
            np.count_nonzero(glare_candidates)
            / np.count_nonzero(interior)
        ),
    })

    rejection_reasons = []
    review_reasons = []

    brightness = metrics["mean_brightness"]
    sharpness = metrics["laplacian_variance"]
    glare_fraction = metrics["glare_candidate_fraction"]

    # Apply exposure rules
    if brightness < config.reject_brightness_low:
        rejection_reasons.append("The image is extremely dark.")
    elif brightness < config.review_brightness_low:
        review_reasons.append("Possible underexposure.")

    if brightness > config.reject_brightness_high:
        rejection_reasons.append("The image is extremely bright.")
    elif brightness > config.review_brightness_high:
        review_reasons.append("Possible overexposure.")

    # Apply sharpness rules
    if sharpness < config.reject_sharpness:
        rejection_reasons.append("The image has extremely low sharpness.")
    elif sharpness < config.review_sharpness:
        review_reasons.append("Low sharpness requires visual review.")

    # Apply contrast and foreground-coverage rules
    if metrics["contrast_std"] < config.review_contrast:
        review_reasons.append("The image has low contrast.")

    if coverage < config.review_coverage:
        review_reasons.append("The visible foreground occupies a small area.")

    # Apply glare-candidate rules
    if glare_fraction > config.reject_glare_fraction:
        rejection_reasons.append(
            "An extensive bright, low-saturation region was detected."
        )
    elif glare_fraction > config.review_glare_fraction:
        review_reasons.append(
            "Bright regions require visual review for possible glare."
        )

    if rejection_reasons:
        status = "rejected"
    elif review_reasons:
        status = "review"
    else:
        status = "accepted"

    return quality_gate_result(
        status,
        rejection_reasons + review_reasons,
        metrics,
    )


