import cv2
from src.function import pixel_art

NUM_CLUSTERS = 10  # Number of colors in the output
PIXEL_SIZE = 4  # Size of each pixel block (e.g., 4x4)
INPUT_IMAGE_PATH = "img/lake.jpg"  # Path to the input image
OUTPUT_IMAGE_PATH = f"output/pixel_{INPUT_IMAGE_PATH.split('/')[-1]}"  # Path to save the output

try:
    # Generate pixel art
    pixelated_image = pixel_art(INPUT_IMAGE_PATH, num_clusters=NUM_CLUSTERS, scale=PIXEL_SIZE)

    # Save and display the result
    cv2.imwrite(OUTPUT_IMAGE_PATH, pixelated_image)  # Save the output image
    cv2.imshow('Pixel Art', pixelated_image)  # Display the result
    cv2.waitKey(0)  # Wait for a key press to close the window
except Exception as e:
    print(f"Error: {str(e)}")  # Handle any errors
finally:
    cv2.destroyAllWindows()  # Clean up OpenCV windows
