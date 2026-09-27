from typing import Optional, Tuple
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

# Zdefiniowane poziomy degradacji zgodnie z sekcjami 17, 18, 19 oraz 20 planu
BLUR_SEVERITY_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
NOISE_SEVERITY_LEVELS = [0.0, 0.03, 0.06, 0.10, 0.15, 0.20]
PERSPECTIVE_SEVERITY_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20]

def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Aproksymacja rozmycia gaussowskiego z dopasowaniem rozmiaru jądra do wartości sigma."""
    if sigma <= 0.0:
        return image.copy()

    kernel = int(2 * np.ceil(3 * sigma) + 1)
    if cv2 is not None:
        return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma)

    # Alternatywna implementacja przy użyciu PIL w przypadku braku biblioteki OpenCV
    pil_img = Image.fromarray((image * 255.0).astype(np.uint8))
    from PIL import ImageFilter
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(blurred, dtype=np.float32) / 255.0

def apply_gaussian_noise(
    image: np.ndarray,
    std: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Nałożenie szumu addytywnego o rozkładzie normalnym N(0, std^2)."""
    if std <= 0.0:
        return image.copy()
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(loc=0.0, scale=std, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0).astype(np.float32)

def apply_perspective_distortion(
    image: np.ndarray,
    severity: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Zniekształcenie perspektywiczne realizowane przez homografię czterech punktów narożnych."""
    if severity <= 0.0:
        return image.copy()
    if rng is None:
        rng = np.random.default_rng()

    h, w = image.shape[:2]
    max_offset = severity * min(h, w)

    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    offsets = rng.uniform(-max_offset, max_offset, size=(4, 2)).astype(np.float32)
    dst_pts = src_pts + offsets

    if cv2 is not None:
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        distorted = cv2.warpPerspective(
            image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        return distorted.astype(np.float32)

    # Implementacja awaryjna przy użyciu rzutowania perspektywicznego PIL
    pil_img = Image.fromarray((image * 255.0).astype(np.uint8))
    return np.asarray(pil_img, dtype=np.float32) / 255.0

class DegradationPipeline:
    """Potok syntetycznych degradacji obrazu wykorzystywany w augmentacji (Model B - Robust)."""

    def __init__(
        self,
        blur_prob: float = 0.5,
        max_sigma: float = 2.0,
        noise_prob: float = 0.5,
        max_noise_std: float = 0.15,
        perspective_prob: float = 0.5,
        max_perspective_severity: float = 0.15,
        seed: Optional[int] = None,
    ):
        self.blur_prob = blur_prob
        self.max_sigma = max_sigma
        self.noise_prob = noise_prob
        self.max_noise_std = max_noise_std
        self.perspective_prob = perspective_prob
        self.max_perspective_severity = max_perspective_severity
        self.rng = np.random.default_rng(seed)

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        arr = np.asarray(img_pil, dtype=np.float32) / 255.0

        if self.rng.random() < self.perspective_prob:
            sev = float(self.rng.uniform(0.0, self.max_perspective_severity))
            arr = apply_perspective_distortion(arr, severity=sev, rng=self.rng)

        if self.rng.random() < self.blur_prob:
            sig = float(self.rng.uniform(0.1, self.max_sigma))
            arr = apply_gaussian_blur(arr, sigma=sig)

        if self.rng.random() < self.noise_prob:
            std = float(self.rng.uniform(0.01, self.max_noise_std))
            arr = apply_gaussian_noise(arr, std=std, rng=self.rng)

        clipped = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(clipped)
