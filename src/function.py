import cv2
import numpy as np
from sklearn.cluster import KMeans


def pixel_art(frame, num_clusters=20, scale=8):
    '''
    frame: The individual frame of a video file (frames list accessed in a for loop)
    num_clusters: The total number of different colors in each frame
    scale: Each "square" in an output frame has scale x scale pixels (scale=8 means 8x8 pixels)

    '''

    img = cv2.imread(frame)
    # Combine the height and width dimensions so that kmeans can read the data
    flattened_img = img.reshape([int(img.shape[0] * img.shape[1]), img.shape[2]])
    # Initialize and fit kmeans clustering algorithm
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(flattened_img)
    final_clusters = np.round(kmeans.cluster_centers_).astype(np.uint8)

    # Get the rescaled frame dimensions depending on the scale
    frame_resized = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)))
    # Combine the height and width dimensions so that kmeans can read the data
    resized_flattened = frame_resized.reshape(
        [int(frame_resized.shape[0] * frame_resized.shape[1]), frame_resized.shape[2]]
    )
    clusters = kmeans.predict(resized_flattened)
    new_img = final_clusters[clusters].reshape(frame_resized.shape) / 255

    canvas = np.empty(img.shape)
    # Paint the canvas with the correct cluster point
    for y in range(frame_resized.shape[0]):
        for x in range(frame_resized.shape[1]):
            ix = int(scale * x)
            iy = int(scale * y)
            ix_ = int(scale * (x + 1))
            iy_ = int(scale * (y + 1))
            color = new_img[y, x]
            canvas[iy:iy_, ix:ix_] = color

    return canvas