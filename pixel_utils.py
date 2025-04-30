import cv2
import numpy as np
from sklearn.cluster import KMeans


def generate_pixel_art(
        image: np.ndarray,
        num_colors: int = 8,
        pixel_size: int = 8,
        enable_outline: bool = False
) -> np.ndarray:
    """
    Convert an input image to pixel art style

    Args:
        image: Input image in BGR format
        num_colors: Number of colors for quantization (4-32)
        pixel_size: Size of each pixel block (4-32)
        enable_outline: Whether to add retro game-style outlines

    Returns:
        Pixel art image in BGR format
    """
    # Validate input
    if image is None:
        raise ValueError("Invalid input image")
    if pixel_size < 1:
        raise ValueError("Pixel size must be ≥1")

    # Color quantization
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=0)
    kmeans.fit(pixels)
    color_palette = kmeans.cluster_centers_.astype(np.uint8)

    # Downsampling
    downsampled = cv2.resize(
        image,
        (image.shape[1] // pixel_size, image.shape[0] // pixel_size),
        interpolation=cv2.INTER_NEAREST
    )
    color_labels = kmeans.predict(downsampled.reshape(-1, 3))
    quantized_image = color_palette[color_labels].reshape(downsampled.shape)

    # Pixel block reconstruction
    canvas = np.zeros_like(image)
    h, w = downsampled.shape[:2]
    for y in range(h):
        for x in range(w):
            y_start = y * pixel_size
            y_end = (y + 1) * pixel_size
            x_start = x * pixel_size
            x_end = (x + 1) * pixel_size
            canvas[y_start:y_end, x_start:x_end] = quantized_image[y, x]

    # Outline enhancement
    if enable_outline:
        grayscale = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grayscale, threshold1=50, threshold2=150)
        canvas[edges != 0] = [0, 0, 0]  # Black outlines

    return canvas
