"""Local upload evidence for the notebook's retinal preprocessing techniques."""
from __future__ import annotations

import base64
import csv
from dataclasses import asdict, replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import time
import sys

import cv2
import numpy as np
from PIL import Image
from skimage.filters import frangi

from dr_preprocessing import (
    PIPELINE_VERSION, PreprocessConfig, read_rgb, extract_retina, resize_pad,
    denoise_image, illumination_correct, clahe_green, enhance_edges,
    normalise_rgb,
)

ROOT = Path(__file__).resolve().parent
CONFIG = PreprocessConfig()
MODEL_CONFIG = json.loads((ROOT / 'model_input_config.json').read_text(encoding='utf-8'))['resolved_defaults']
LABELS = {}
with (ROOT / 'tables' / 'saved_splits.csv').open(newline='', encoding='utf-8') as handle:
    for row in csv.DictReader(handle):
        LABELS[row['id_code']] = int(row['diagnosis'])

_CACHE: dict[str, dict] = {}


def visual(label: str, image: np.ndarray) -> dict:
    return {'label': label, 'pixels': image}


def step(number: int, title: str, purpose: str, metrics: dict, images: list[dict] | None = None, note: str = '') -> dict:
    return {'number': number, 'title': title, 'purpose': purpose, 'metrics': metrics,
            'note': note, '_visuals': images or []}


def encode_preview(pixels: np.ndarray) -> str:
    image_pixels = np.asarray(pixels)
    if image_pixels.dtype == np.bool_:
        image_pixels = image_pixels.astype(np.uint8) * 255
    elif np.issubdtype(image_pixels.dtype, np.floating):
        image_pixels = np.nan_to_num(image_pixels)
        if image_pixels.size and image_pixels.min() >= 0 and image_pixels.max() <= 1:
            image_pixels = image_pixels * 255
        image_pixels = np.clip(image_pixels, 0, 255).astype(np.uint8)
    else:
        image_pixels = np.clip(image_pixels, 0, 255).astype(np.uint8)

    if image_pixels.ndim == 2:
        image_pixels = cv2.cvtColor(image_pixels, cv2.COLOR_GRAY2RGB)
    if image_pixels.ndim != 3 or image_pixels.shape[2] != 3:
        raise ValueError(f'Cannot render preprocessing preview with shape {image_pixels.shape}.')

    image = Image.fromarray(image_pixels)
    image.thumbnail((240, 180), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=72, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{encoded}'


def masked(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.copy()
    result[~mask] = 0
    return result


def difference(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return round(float(np.mean(np.abs(a[mask].astype(np.float32) - b[mask].astype(np.float32)))), 2) if mask.any() else 0.0


def glare_masks(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    candidates = ((hsv[..., 2] >= 230) & (hsv[..., 1] <= 40) & mask).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    small = np.zeros(mask.shape, np.uint8)
    maximum = max(1, int(mask.sum() * .001))
    interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    for index in range(1, count):
        region = components == index
        if int(stats[index, cv2.CC_STAT_AREA]) <= maximum and np.all(interior[region]):
            small[region] = 255
    return candidates * 255, small


def lab_alternative(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness = lab[..., 0]
    blur = cv2.GaussianBlur(lightness, (0, 0), CONFIG.illumination_sigma)
    corrected = np.clip(4 * lightness.astype(np.float32) - 4 * blur.astype(np.float32) + 128, 0, 255).astype(np.uint8)
    lab[..., 0] = cv2.createCLAHE(clipLimit=CONFIG.clahe_clip, tileGridSize=CONFIG.clahe_grid).apply(corrected)
    return masked(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), mask)


def gaussian_division(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    weights = mask.astype(np.float32)
    denominator = cv2.GaussianBlur(weights, (0, 0), 30)
    numerator = cv2.GaussianBlur(values * weights[..., None], (0, 0), 30)
    background = numerator / np.maximum(denominator[..., None], 1e-6)
    median = np.median(values[mask], axis=0)
    corrected = values / np.maximum(background, 1.0) * median
    corrected *= median / np.maximum(np.median(corrected[mask], axis=0), 1e-6)
    return masked(np.clip(corrected, 0, 255).astype(np.uint8), mask)


def vessel_map(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    green = image[..., 1].astype(np.float32) / 255
    tissue_median = float(np.median(green[mask]))
    response = frangi(np.where(mask, green, tissue_median), sigmas=(1, 2, 3), black_ridges=True, mode='reflect')
    response = np.nan_to_num(response).astype(np.float32)
    interior = cv2.erode(mask.astype(np.uint8), np.ones((19, 19), np.uint8)) > 0
    response *= interior
    if response.max() > 0:
        response /= response.max()
    return response


def assess_quality(image: np.ndarray, mask: np.ndarray, glare: np.ndarray, crop_info: dict) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if interior.sum() < 32:
        interior = mask
    if interior.sum() < 32:
        return {'status': 'rejected', 'reasons': ['Insufficient visible image region.'], 'metrics': {}}
    values = gray[interior].astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    metrics = {
        'mean brightness': round(float(values.mean()), 2),
        'contrast std': round(float(values.std()), 2),
        'sharpness variance': round(float(laplacian[interior].var()), 2),
        'dark fraction': round(float((values < 15).mean()), 4),
        'bright fraction': round(float((values > 245).mean()), 4),
        'glare fraction': round(float(np.count_nonzero(glare) / max(1, np.count_nonzero(mask))), 4),
        'mask fraction': round(float(mask.mean()), 4),
    }
    rejected, review = [], []
    if metrics['mean brightness'] < 15: rejected.append('Extremely dark image.')
    if metrics['mean brightness'] > 240: rejected.append('Extremely bright image.')
    if metrics['glare fraction'] > .15: rejected.append('Extensive bright/glare candidate region.')
    if crop_info['mask_fallback']: review.append('Retinal extraction was inconclusive.')
    if metrics['mean brightness'] < 35: review.append('Possible underexposure.')
    if metrics['mean brightness'] > 200: review.append('Possible overexposure.')
    if metrics['sharpness variance'] < 50: review.append('Low sharpness score.')
    if metrics['contrast std'] < 15: review.append('Low contrast.')
    if metrics['glare fraction'] > .05: review.append('Bright regions require visual review.')
    return {'status': 'rejected' if rejected else 'review' if review else 'passed', 'reasons': rejected + review, 'metrics': metrics}


BRANCHES = {'baseline', 'denoise', 'ben', 'clahe', 'unsharp', 'morphology', 'lab', 'gaussian', 'inpaint'}


def analyze(payload: bytes, filename: str, known_grade: int | None = None, branch: str = 'baseline',
            include_visuals: bool = False) -> dict:
    if not payload:
        raise ValueError('Choose a JPEG or PNG image first.')
    if len(payload) > CONFIG.max_file_mb * 1024 * 1024:
        raise ValueError('File exceeds the configured size limit.')
    if branch not in BRANCHES:
        raise ValueError('Unknown reconstruction branch.')
    sha = hashlib.sha256(payload).hexdigest()
    cache_key = hashlib.sha256((sha + Path(filename).name + str(known_grade) + branch + json.dumps(asdict(CONFIG), sort_keys=True) + PIPELINE_VERSION).encode()).hexdigest()
    if cache_key in _CACHE and not include_visuals:
        cached = json.loads(json.dumps(_CACHE[cache_key]))
        cached['cache_hit'] = True
        cached['steps'][12]['metrics']['cache'] = 'hit'
        return cached
    started = time.perf_counter()
    suffix = Path(filename).suffix.lower()
    with tempfile.TemporaryDirectory() as folder:
        temp_path = Path(folder) / ('upload' + suffix)
        temp_path.write_bytes(payload)
        try:
            original, input_info = read_rgb(temp_path, CONFIG)
        except (ValueError, OSError, Image.UnidentifiedImageError) as exc:
            raise ValueError(f'File validation failed: {exc}') from exc
    basename = Path(filename).name
    manifest_grade = LABELS.get(Path(basename).stem)
    if known_grade is not None and manifest_grade is not None and known_grade != manifest_grade:
        raise ValueError(f'Grade mismatch: the saved APTOS manifest labels this image grade {manifest_grade}.')
    label_check = ('matches saved APTOS manifest' if known_grade is not None and manifest_grade is not None else
                   'manifest label available' if manifest_grade is not None else
                   'user-provided, not independently verified' if known_grade is not None else 'not supplied')
    steps = [step(1, 'File validation', 'Decode, orient and convert the JPEG/PNG to RGB; check dimensions and any supplied grade.',
                  {'format': input_info['format'], 'dimensions': f"{input_info['width']} × {input_info['height']}",
                   'bytes': input_info['file_bytes'], 'grade check': label_check}, [visual('Validated RGB', original)])]
    crop, crop_mask, full_mask, crop_info = extract_retina(original, CONFIG)
    steps.append(step(2, 'Retinal extraction', 'Detect the outer retinal boundary, retain dark internal structure and crop excess background.',
                      {'mask coverage': f"{crop_info['candidate_mask_fraction']*100:.1f}%", 'crop (x0,y0,x1,y1)': crop_info['bbox_xyxy'], 'mask fallback': crop_info['mask_fallback']},
                      [visual('Cropped retina', crop), visual('Retinal mask', full_mask.astype(np.uint8)*255)]))
    base, mask, geometry = resize_pad(crop, crop_mask, CONFIG.size)
    steps.append(step(3, 'Resize and padding', 'Preserve aspect ratio; fit the longest side to 224 and pad evenly.',
                      {'resized height × width': geometry['resized_hw'], 'top,left padding': geometry['padding_top_left'], 'scale': round(geometry['scale'], 4)},
                      [visual('224 × 224 RGB', base), visual('Resized mask', mask.astype(np.uint8)*255)]))
    denoised = denoise_image(base, mask, replace(CONFIG, denoise=True))
    steps.append(step(4, 'Denoising', 'Compare mild bilateral filtering against the unfiltered image.',
                      {'mean absolute change': difference(base, denoised, mask)}, [visual('No filtering', base), visual('Bilateral', denoised)], 'Visual inspection is needed to judge any lost small lesion detail.'))
    ben = illumination_correct(base, mask, CONFIG)
    saturation = float(np.mean((ben[mask] == 0) | (ben[mask] == 255)))
    steps.append(step(5, 'Ben Graham correction', 'Test local illumination correction and inspect saturation and boundary artefacts.',
                      {'saturated channel fraction': round(saturation, 4), 'mean absolute change': difference(base, ben, mask)}, [visual('Original', base), visual('Corrected', ben)]))
    clahe = clahe_green(base, mask, CONFIG)
    steps.append(step(6, 'Green-channel CLAHE', 'Test local contrast enhancement on the green channel.',
                      {'green std before': round(float(base[...,1][mask].std()),2), 'green std after': round(float(clahe[...,1][mask].std()),2)}, [visual('Original', base), visual('CLAHE', clahe)], 'Contrast increase can also amplify noise; inspect the image.'))
    unsharp = enhance_edges(base, mask, replace(CONFIG, edge='unsharp'))
    morph = enhance_edges(base, mask, replace(CONFIG, edge='morphology'))
    steps.append(step(7, 'Edge enhancement', 'Compare mild unsharp masking with top-hat/black-hat morphology.',
                      {'unsharp mean change': difference(base, unsharp, mask), 'morphology mean change': difference(base, morph, mask)},
                      [visual('Original', base), visual('Unsharp', unsharp), visual('Top-hat / black-hat', morph)], 'Inspect for halos or artificial detail before selecting an enhancement.'))
    lab = lab_alternative(base, mask)
    steps.append(step(8, 'LAB alternative', 'Enhance lightness separately from chromatic channels and compare with green-channel CLAHE.',
                      {'LAB mean change': difference(base, lab, mask)}, [visual('Original', base), visual('LAB lightness', lab), visual('Green CLAHE', clahe)]))
    glare, repair = glare_masks(base, mask)
    inpainted = cv2.inpaint(base, repair, 3, cv2.INPAINT_NS) if repair.any() else base.copy()
    steps.append(step(9, 'Glare detection', 'Find bright low-saturation candidates; optionally inpaint small interior spots.',
                      {'candidate fraction': round(float(np.count_nonzero(glare)/max(1,mask.sum())),4), 'inpaint pixels': int(np.count_nonzero(repair))},
                      [visual('Original', base), visual('Candidates', glare), visual('Optional inpaint', inpainted)]))
    division = gaussian_division(base, mask)
    steps.append(step(10, 'Gaussian division', 'Compare mask-aware shading correction with Ben Graham correction.',
                      {'division mean change': difference(base, division, mask)}, [visual('Original', base), visual('Division', division), visual('Ben Graham', ben)]))
    vessel = vessel_map(base, mask)
    steps.append(step(11, 'Frangi maps', 'Generate a separate multiscale vessel-response map; keep classifier input as three-channel RGB.',
                      {'scales': '1, 2, 3 pixels', 'classifier channels': 3,
                       'mean vessel response': round(float(vessel[mask].mean()), 4),
                       'strong response fraction': round(float((vessel[mask] > .5).mean()), 4)},
                      [visual('RGB', base), visual('Vessel response', vessel)]))
    branch_images = {'baseline': base, 'denoise': denoised, 'ben': ben, 'clahe': clahe,
                     'unsharp': unsharp, 'morphology': morph, 'lab': lab,
                     'gaussian': division, 'inpaint': inpainted}
    final = masked(branch_images[branch], mask)
    if final.shape != (224,224,3) or final.dtype != np.uint8:
        raise ValueError('Final RGB integrity check failed.')
    steps.append(step(12, 'RGB reconstruction', 'Clip and mask the selected RGB branch, then verify channel order, dimensions and data type.',
                      {'shape': '224 × 224 × 3', 'dtype': str(final.dtype), 'channel order': 'RGB', 'selected branch': branch},
                      [visual('Selected reconstructed RGB', final)],
                      'Baseline matches the recorded model setup. Other branches are visual alternatives and were not validated for this checkpoint.'))
    elapsed = round(time.perf_counter()-started,3)
    steps.append(step(13, 'Reproducibility and caching', 'Fingerprint the input and selected parameters; cache deterministic results before training augmentation.',
                      {'input SHA-256': sha[:16]+'…', 'pipeline version': PIPELINE_VERSION,
                       'size': CONFIG.size, 'mask threshold': CONFIG.mask_threshold,
                       'selected branch': branch, 'Python': sys.version.split()[0],
                       'NumPy': np.__version__, 'OpenCV': cv2.__version__,
                       'cache': 'miss', 'processing seconds': elapsed},
                      [visual('Baseline model image', base)],
                      note='Training-only random augmentation is not applied to this inference walkthrough.'))
    quality = assess_quality(base, mask, glare, crop_info)
    steps.append(step(14, 'Quality assessment', 'Measure the unenhanced image independently of optional enhancements.',
                      {'decision': quality['status'], **quality['metrics']}, [visual('Unenhanced image', base), visual('Glare candidates', glare)],
                      ' '.join(quality['reasons']) or 'No quality flags from these unvalidated heuristics.'))
    tensor = normalise_rgb(final, MODEL_CONFIG['mean'], MODEL_CONFIG['std'])
    steps.append(step(15, 'Backbone normalisation', 'Apply pretrained-model channel mean and standard deviation exactly once after preprocessing.',
                      {'tensor shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'range': [round(float(tensor.min()),3),round(float(tensor.max()),3)], 'mean': MODEL_CONFIG['mean'], 'std': MODEL_CONFIG['std']},
                      [visual('RGB image before normalization', final)],
                      note='The tensor is prepared for EfficientNetB0. Only the baseline branch reflects the recorded model input setup; this walkthrough does not run the trained checkpoint.'))
    for item in steps:
        visuals = item.pop('_visuals')
        if include_visuals:
            item['images'] = [
                {'label': preview['label'], 'data_url': encode_preview(preview['pixels'])}
                for preview in visuals
            ]
    result = {'filename': basename, 'quality': quality, 'grade_available': quality['status'] != 'rejected',
              'grade_message': 'Image rejected by quality gate; grading bypassed.' if quality['status']=='rejected' else 'Image prepared for model inference; grading is not run in this walkthrough.',
              'steps': steps, 'cache_hit': False}
    if not include_visuals:
        if len(_CACHE) >= 8:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[cache_key] = result
    return result
