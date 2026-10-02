"""Conservative visual input guard for colour fundus photographs.

This is not a validated retinal-image classifier. Uncertain images are blocked.
"""

import cv2
import numpy as np


def assess_retinal_appearance(rgb):
    height, width = rgb.shape[:2]
    scale = 384 / max(height, width)
    small = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    corner_size = max(4, round(min(h, w) * .12))
    corners = [gray[:corner_size, :corner_size], gray[:corner_size, -corner_size:],
               gray[-corner_size:, :corner_size], gray[-corner_size:, -corner_size:]]
    dark_corners = sum(float(np.median(corner)) < 50 for corner in corners)

    foreground = np.uint8(gray > 20) * 255
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea) if contours else None
    mask = np.zeros_like(gray, dtype=np.uint8)
    if largest is not None:
        cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)
    coverage = float(np.mean(mask > 0))
    area = float(cv2.contourArea(largest)) if largest is not None else 0.0
    perimeter = float(cv2.arcLength(largest, True)) if largest is not None else 0.0
    circularity = 4 * np.pi * area / perimeter**2 if perimeter else 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(largest))) if largest is not None else 0.0
    solidity = area / hull_area if hull_area else 0.0

    interior = cv2.erode(mask, np.ones((7, 7), np.uint8)) > 0
    pixels = small[interior] if np.any(interior) else small.reshape(-1, 3)
    red, green, blue = np.median(pixels, axis=0).tolist()
    # Thin, darker structure is expected within a fundus image. This also
    # rejects featureless coloured discs and many busy, non-medical photos.
    green_channel = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(small[..., 1])
    blackhat = cv2.morphologyEx(
        green_channel, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    texture_density = float(np.mean(blackhat[interior] > 12)) if np.any(interior) else 0.0

    signals = {
        'dark_surround': dark_corners >= 2,
        'retinal_coverage': .30 <= coverage <= .90,
        'rounded_boundary': circularity >= .82 and solidity >= .90,
        'fundus_colour': red >= green * 1.20 and red >= blue * 1.40,
        'fine_structure': .045 <= texture_density <= .45,
    }
    return {
        'likely_retinal': all(signals.values()),
        'method': 'Conservative, unvalidated fundus-photo appearance gate',
        'signals': signals,
        'measurements': {
            'dark_corners': dark_corners,
            'foreground_coverage': round(coverage, 3),
            'circularity': round(circularity, 3),
            'solidity': round(solidity, 3),
            'red_green_ratio': round(red / max(green, 1), 3),
            'red_blue_ratio': round(red / max(blue, 1), 3),
            'fine_structure_density': round(texture_density, 3),
        },
    }
