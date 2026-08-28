import cv2
import numpy as np
from skimage.measure import shannon_entropy

FEATURE_NAMES = [
    "laplacian_var",
    "laplacian_mean_abs",
    "sobel_mean",
    "sobel_var",
    "grad_mag_mean",
    "brightness_mean",
    "brightness_median",
    "brightness_p5",
    "brightness_p95",
    "brightness_std",
    "dark_ratio",
    "bright_ratio",
    "contrast_p95_p5",
    "saturation_mean",
    "saturation_std",
    "entropy",
    "edge_density",
    "noise_est",
    "blockiness",
    "hf_ratio",
]

def _blockiness(gray: np.ndarray) -> float:
    # Measure 8x8 block boundary differences vs interior (JPEG blockiness proxy)
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0
    # Horizontal boundaries
    # align sizes: truncate to min rows/cols
    rows_h = min(gray[7::8, :].shape[0], gray[8::8, :].shape[0])
    cols_v = min(gray[:, 7::8].shape[1], gray[:, 8::8].shape[1])
    diff_h = np.abs(gray[7::8, :][:rows_h] - gray[8::8, :][:rows_h]).mean() if h >= 16 and rows_h>0 else 0
    diff_v = np.abs(gray[:, 7::8][:,:cols_v] - gray[:, 8::8][:,:cols_v]).mean() if w >= 16 and cols_v>0 else 0
    # Interior diff
    interior = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    if interior < 1e-6:
        return 0.0
    return float(((diff_h + diff_v) / 2.0) / (interior + 1e-6) - 1.0)

def extract_features(rgb: np.ndarray, gray: np.ndarray) -> dict:
    # Ensure float
    gray_f = gray.astype(np.float32)
    rgb_f = rgb.astype(np.float32) if rgb.ndim == 3 else np.stack([gray_f]*3, axis=-1)

    # Sharpness
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())
    lap_mean_abs = float(np.abs(lap).mean())

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mean = float(sobel_mag.mean())
    sobel_var = float(sobel_mag.var())

    grad_mag_mean = sobel_mean  # duplicate for clarity

    # Brightness stats (luminance = gray)
    brightness_mean = float(gray_f.mean())
    brightness_median = float(np.median(gray_f))
    brightness_p5 = float(np.percentile(gray_f, 5))
    brightness_p95 = float(np.percentile(gray_f, 95))
    brightness_std = float(gray_f.std())
    dark_ratio = float((gray_f < 30).mean())
    bright_ratio = float((gray_f > 225).mean())
    contrast_p95_p5 = float(brightness_p95 - brightness_p5)

    # Saturation (HSV)
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    saturation_mean = float(sat.mean())
    saturation_std = float(sat.std())

    # Entropy
    try:
        entropy = float(shannon_entropy(gray))
    except:
        entropy = 0.0

    # Edge density (Canny)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((edges > 0).mean())

    # Noise estimate: use Laplacian residual method (Immerkaer)
    # Fast estimate: variance of Laplacian response adjusted
    # noise = std of high-pass filtered image
    # Use 3x3 Laplacian kernel variance
    lap_noise = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    # Robust sigma via MAD
    mad = np.median(np.abs(lap_noise - np.median(lap_noise)))
    noise_est = float(1.4826 * mad / 6.0)  # scaling for Laplacian
    # Alternative: cap
    noise_est = float(np.clip(noise_est, 0, 50))

    blockiness = float(_blockiness(gray))
    blockiness = float(np.clip(blockiness, -1, 5))

    # High frequency ratio: FFT energy above threshold
    try:
        f = np.fft.fft2(gray_f)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        h, w = gray.shape
        cy, cx = h//2, w//2
        # High frequency = outside central 25%
        rh, rw = h//4, w//4
        low = mag[cy-rh:cy+rh, cx-rw:cx+rw].sum()
        total = mag.sum() + 1e-9
        hf_ratio = float((total - low) / total)
    except:
        hf_ratio = 0.0

    feats = {
        "laplacian_var": lap_var,
        "laplacian_mean_abs": lap_mean_abs,
        "sobel_mean": sobel_mean,
        "sobel_var": sobel_var,
        "grad_mag_mean": grad_mag_mean,
        "brightness_mean": brightness_mean,
        "brightness_median": brightness_median,
        "brightness_p5": brightness_p5,
        "brightness_p95": brightness_p95,
        "brightness_std": brightness_std,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "contrast_p95_p5": contrast_p95_p5,
        "saturation_mean": saturation_mean,
        "saturation_std": saturation_std,
        "entropy": entropy,
        "edge_density": edge_density,
        "noise_est": noise_est,
        "blockiness": blockiness,
        "hf_ratio": hf_ratio,
    }
    return feats

def features_to_vector(feats: dict) -> np.ndarray:
    return np.array([feats[n] for n in FEATURE_NAMES], dtype=np.float32)
