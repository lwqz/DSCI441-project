import cv2
import numpy as np
from sklearn.cluster import KMeans


def pixel_art(image_path: str, num_clusters: int = 20, scale: int = 8) -> np.ndarray:
    """
    Generate pixel art from an input image.

    Args:
        image_path (str): Path to the input image.
        num_clusters (int): Number of colors for quantization (default: 20).
        scale (int): Size of each pixel block (default: 8x8).

    Returns:
        np.ndarray: Pixel art image in BGR format.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Color quantization using K-means
    flattened_img = img.reshape(-1, 3)  # Flatten to (height*width, 3)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)  # Initialize K-means
    kmeans.fit(flattened_img)  # Fit the model
    final_clusters = np.round(kmeans.cluster_centers_).astype(np.uint8)  # Get color palette

    # Downsample the image
    frame_resized = cv2.resize(img,
                             (img.shape[1] // scale, img.shape[0] // scale),
                             interpolation=cv2.INTER_NEAREST)  # Nearest-neighbor interpolation
    resized_flattened = frame_resized.reshape(-1, 3)  # Flatten the resized image
    clusters = kmeans.predict(resized_flattened)  # Predict color labels
    new_img = final_clusters[clusters].reshape(frame_resized.shape)  # Map labels to colors

    # Create the pixel art canvas
    canvas = np.zeros_like(img, dtype=np.uint8)  # Initialize canvas with the same size as input
    for y in range(frame_resized.shape[0]):
        for x in range(frame_resized.shape[1]):
            x_start = scale * x  # Start x-coordinate of the block
            y_start = scale * y  # Start y-coordinate of the block
            x_end = scale * (x + 1)  # End x-coordinate of the block
            y_end = scale * (y + 1)  # End y-coordinate of the block
            color = new_img[y, x].astype(np.uint8)  # Get the quantized color
            canvas[y_start:y_end, x_start:x_end] = color  # Fill the block with the color

    return canvas
