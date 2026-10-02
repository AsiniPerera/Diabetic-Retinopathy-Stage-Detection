# PIPELINE_EXPORT validation
from pathlib import Path
from dataclasses import dataclass, asdict, replace
import time
import warnings
import numpy as np
import cv2
from PIL import Image, ImageOps

PIPELINE_VERSION = '1.0.0'

@dataclass(frozen=True)
class PreprocessConfig:
    size: int = 224
    max_file_mb: int = 128
    max_pixels: int = 80_000_000
    min_side: int = 32
    mask_threshold: int = 8
    mask_min_fraction: float = 0.03
    crop_margin: int = 4
    denoise: bool = False
    bilateral_d: int = 5
    bilateral_sigma_color: float = 20.0
    bilateral_sigma_space: float = 20.0
    illumination: bool = False
    illumination_sigma: float = 7.0
    illumination_a: float = 4.0
    illumination_b: float = 4.0
    illumination_c: float = 128.0
    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_grid: tuple = (8, 8)
    edge: str = 'none'  # none, unsharp, morphology
    unsharp_amount: float = 0.3
    unsharp_sigma: float = 1.0
    morph_kernel: int = 5
    morph_amount: float = 0.2

def read_rgb(path, config):
    """Validate the decoded image, then return RGB uint8 pixels and input metadata."""
    path = Path(path)
    if not path.is_file():
        raise ValueError('File does not exist.')
    if path.stat().st_size > config.max_file_mb * 1024**2:
        raise ValueError('File exceeds the configured size limit.')
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(path) as im:
            detected_format = im.format
            original_mode = im.mode
            if detected_format not in {'JPEG','PNG'}:
                raise ValueError('Decoded format is not JPEG or PNG.')
            if min(im.size) < config.min_side:
                raise ValueError('Image is smaller than the operational minimum size.')
            if im.width * im.height > config.max_pixels:
                raise ValueError('Image exceeds the configured pixel limit.')
            if getattr(im, 'n_frames', 1) != 1:
                raise ValueError('Animated or multi-frame input is unsupported.')
            im.load()
            oriented = ImageOps.exif_transpose(im)
            if 'A' in oriented.getbands() or 'transparency' in oriented.info:
                rgba = oriented.convert('RGBA')
                black = Image.new('RGBA', rgba.size, (0,0,0,255))
                oriented = Image.alpha_composite(black, rgba)
            rgb = np.array(oriented.convert('RGB'), dtype=np.uint8)
    return rgb, {'format': detected_format, 'original_mode': original_mode,
                 'height': int(rgb.shape[0]), 'width': int(rgb.shape[1]),
                 'file_bytes': path.stat().st_size}

# PIPELINE_EXPORT geometry
def extract_retina(rgb, config):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    small_factor = min(1.0, 768 / max(h,w))
    sw, sh = max(1,round(w*small_factor)), max(1,round(h*small_factor))
    small = cv2.resize(gray, (sw,sh), interpolation=cv2.INTER_AREA)
    binary = (small > config.mask_threshold).astype(np.uint8)*255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate = np.zeros_like(binary)
    if contours:
        cv2.drawContours(candidate, [max(contours,key=cv2.contourArea)], -1, 255, -1)
    mask = cv2.resize(candidate, (w,h), interpolation=cv2.INTER_NEAREST) > 0
    area_fraction = float(mask.mean())
    fallback = not contours or area_fraction < config.mask_min_fraction
    if fallback:
        mask = np.ones((h,w), dtype=bool)
        bbox = (0,0,w,h)
    else:
        ys,xs = np.where(mask)
        margin = config.crop_margin
        bbox = (max(0,int(xs.min())-margin), max(0,int(ys.min())-margin),
                min(w,int(xs.max())+1+margin), min(h,int(ys.max())+1+margin))
    x0,y0,x1,y1 = bbox
    cropped = rgb[y0:y1,x0:x1].copy()
    cropped_mask = mask[y0:y1,x0:x1]
    cropped[~cropped_mask] = 0
    metadata = {'mask_fallback':bool(fallback), 'candidate_mask_fraction':area_fraction,
                'bbox_xyxy':list(bbox), 'original_shape':list(rgb.shape)}
    return cropped, cropped_mask, mask, metadata

def resize_pad(rgb, mask, size):
    h,w = rgb.shape[:2]
    factor = size/max(h,w)
    nh,nw = max(1,round(h*factor)), max(1,round(w*factor))
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (nw,nh), interpolation=interpolation)
    resized_mask = cv2.resize(mask.astype(np.uint8),(nw,nh),interpolation=cv2.INTER_NEAREST)>0
    top,left = (size-nh)//2, (size-nw)//2
    canvas = np.zeros((size,size,3),np.uint8)
    canvas_mask = np.zeros((size,size),bool)
    canvas[top:top+nh,left:left+nw] = resized
    canvas_mask[top:top+nh,left:left+nw] = resized_mask
    canvas[~canvas_mask] = 0
    geometry = {'scale':factor, 'resized_hw':[nh,nw], 'padding_top_left':[top,left],
                'interpolation':'area' if factor<1 else 'cubic'}
    return canvas, canvas_mask, geometry

# PIPELINE_EXPORT quality
def quality_measurements(unenhanced, resized_mask, original_mask, crop_meta):
    if crop_meta['mask_fallback']:
        return {'quality_status':'mask_failed_measurements_unavailable',
                'sharpness_laplacian_variance':None, 'mean_intensity':None,
                'dark_pixel_fraction':None, 'bright_pixel_fraction':None,
                'retinal_frame_fraction':None, 'centre_offset_fraction':None,
                'frame_edges_touched':None}
    gray = cv2.cvtColor(unenhanced, cv2.COLOR_RGB2GRAY)
    interior = cv2.erode(resized_mask.astype(np.uint8), np.ones((5,5),np.uint8))>0
    if interior.sum() < 20:
        interior = resized_mask
    values = gray[interior].astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    ys,xs = np.where(original_mask)
    h,w = original_mask.shape
    offset = np.hypot((xs.mean()-(w-1)/2)/w, (ys.mean()-(h-1)/2)/h)
    edges = sum(bool(edge.any()) for edge in
                [original_mask[0,:],original_mask[-1,:],original_mask[:,0],original_mask[:,-1]])
    return {'quality_status':'measured_not_clinically_classified',
            'sharpness_laplacian_variance':float(laplacian[interior].var()),
            'mean_intensity':float(values.mean()),
            'dark_pixel_fraction':float((values<20).mean()),
            'bright_pixel_fraction':float((values>235).mean()),
            'retinal_frame_fraction':float(original_mask.mean()),
            'centre_offset_fraction':float(offset), 'frame_edges_touched':int(edges)}

# PIPELINE_EXPORT enhancements
def denoise_image(rgb, mask, config):
    result = cv2.bilateralFilter(rgb, config.bilateral_d,
                                config.bilateral_sigma_color, config.bilateral_sigma_space)
    result[~mask] = 0
    return result

def illumination_correct(rgb, mask, config):
    src = rgb.astype(np.float32)
    weights = mask.astype(np.float32)
    denominator = cv2.GaussianBlur(weights,(0,0),config.illumination_sigma)
    numerator = cv2.GaussianBlur(src*weights[...,None],(0,0),config.illumination_sigma)
    local_mean = numerator/np.maximum(denominator[...,None],1e-6)
    result = np.clip(config.illumination_a*src-config.illumination_b*local_mean+
                     config.illumination_c,0,255).astype(np.uint8)
    result[~mask] = 0
    return result

def clahe_green(rgb, mask, config):
    result = rgb.copy()
    clahe_operator = cv2.createCLAHE(clipLimit=config.clahe_clip,tileGridSize=tuple(config.clahe_grid))
    result[:,:,1] = clahe_operator.apply(rgb[:,:,1])
    result[~mask] = 0
    return result

def enhance_edges(rgb, mask, config):
    result = rgb.copy()
    if config.edge == 'none':
        return result
    green = rgb[:,:,1].astype(np.float32)
    if config.edge == 'unsharp':
        blurred = cv2.GaussianBlur(green,(0,0),config.unsharp_sigma)
        enhanced = green + config.unsharp_amount*(green-blurred)
    elif config.edge == 'morphology':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(config.morph_kernel,config.morph_kernel))
        top = cv2.morphologyEx(green,cv2.MORPH_TOPHAT,kernel)
        black = cv2.morphologyEx(green,cv2.MORPH_BLACKHAT,kernel)
        enhanced = green + config.morph_amount*top - config.morph_amount*black
    else:
        raise ValueError('Unknown edge method.')
    # Keep a boundary band unchanged to reduce artificial mask-edge enhancement.
    interior = cv2.erode(mask.astype(np.uint8),np.ones((7,7),np.uint8))>0
    channel = result[:,:,1]
    channel[interior] = np.clip(enhanced[interior],0,255).astype(np.uint8)
    result[~mask] = 0
    return result

# PIPELINE_EXPORT normalisation
def normalise_rgb(rgb, mean, std):
    """Return float32 CHW array. Do not save this as an ordinary PNG."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected RGB uint8 image.')
    mean = np.asarray(mean,dtype=np.float32)
    std = np.asarray(std,dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or (std<=0).any():
        raise ValueError('Expected 3-channel mean and positive standard deviation.')
    result = (rgb.astype(np.float32)/255.0-mean)/std
    return np.ascontiguousarray(result.transpose(2,0,1),dtype=np.float32)

# PIPELINE_EXPORT pipeline
def preprocess_image(path, config):
    start = time.perf_counter()
    rgb,input_info = read_rgb(path,config)
    crop,crop_mask,full_mask,crop_info = extract_retina(rgb,config)
    base,mask,resize_info = resize_pad(crop,crop_mask,config.size)
    quality = quality_measurements(base,mask,full_mask,crop_info)
    stages = {'original':rgb,'cropped':crop,'resized_unenhanced':base.copy()}
    current = base.copy()
    if config.denoise:
        current = denoise_image(current,mask,config)
        stages['denoised'] = current.copy()
    if config.illumination:
        current = illumination_correct(current,mask,config)
        stages['illumination'] = current.copy()
    if config.clahe:
        current = clahe_green(current,mask,config)
        stages['clahe'] = current.copy()
    if config.edge != 'none':
        current = enhance_edges(current,mask,config)
        stages['edge'] = current.copy()
    current[~mask] = 0
    stages['final'] = current.copy()
    meta = {'input':input_info,'crop':crop_info,'resize':resize_info,'quality':quality,
            'seconds':time.perf_counter()-start,'pipeline_version':PIPELINE_VERSION}
    return current,mask,stages,meta