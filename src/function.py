import cv2
import numpy as np
from sklearn.cluster import KMeans


def pixel_art(frame, num_clusters=10, scale=4):
    img = cv2.imread(frame)
    if img is None:
        raise FileNotFoundError(f"无法加载图片: {frame}")

    # 颜色聚类
    flattened_img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
    kmeans.fit(flattened_img)
    cluster_colors = np.round(kmeans.cluster_centers_).astype(np.uint8)

    # 缩小 & 量化
    small_img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
    labels = kmeans.predict(small_img.reshape((-1, 3)))
    new_img = cluster_colors[labels].reshape(small_img.shape)

    # 放大 & 还原像素风格
    pixel_art_img = cv2.resize(new_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return pixel_art_img
