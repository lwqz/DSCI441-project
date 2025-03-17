import cv2
from src.function import pixel_art

NUM_CLUSTERS = 10  # Number of colors in the output
BLOCK_SIZE = 4
INPUT_IMAGE_PATH = "img/lake.jpg"
OUTPUT_IMAGE_PATH = f"output/pixel_{INPUT_IMAGE_PATH.split('/')[-1]}"

try:
    # Generate pixel art
    pixelated_image = pixel_art(INPUT_IMAGE_PATH, num_clusters=NUM_CLUSTERS, block_size=BLOCK_SIZE)

    # Save and display the result
    cv2.imwrite(OUTPUT_IMAGE_PATH, pixelated_image)
    cv2.imshow('Pixel Art', pixelated_image)
    cv2.waitKey(0)
except Exception as e:
    print(f"Error: {str(e)}")  # Handle any errors
finally:
    cv2.destroyAllWindows()  # Clean up OpenCV windows
