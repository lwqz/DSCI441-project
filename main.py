import cv2
from src.function import pixel_art

NUM_CLUSTERS = 10
SCALE = 4
IMG_PATH = "img/R.jpg"
OUTPUT_PATH = f"output/pixel_{IMG_PATH.split('/')[-1]}"

canvas = pixel_art(IMG_PATH, num_clusters=NUM_CLUSTERS, scale=SCALE)

cv2.imwrite(OUTPUT_PATH, canvas)
print(f"像素艺术已保存至 {OUTPUT_PATH}")

cv2.imshow('Pixel Art', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
