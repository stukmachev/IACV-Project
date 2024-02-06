# Some experiment with adaptative Threshold
# Not good enough results, but good to mention

import cv2

IMAGE_NAME = "pencilcase_foreground"
IMAGE_EXTENSION = "png"
THRESH_BLOCK_SIZE = 11
THRESH_C = 1

img = cv2.imread(f"{IMAGE_NAME}.{IMAGE_EXTENSION}", cv2.IMREAD_GRAYSCALE)

# Adaptive Mean Thresholding
adaptive_mean_threshold = cv2.adaptiveThreshold(
    img,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    THRESH_BLOCK_SIZE,
    THRESH_C,
)
cv2.imwrite(f"./{IMAGE_NAME}_mean_adaptive.png", adaptive_mean_threshold)