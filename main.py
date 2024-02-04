import cv2
import numpy as np
from matplotlib import pyplot as plt

from pymatting import estimate_alpha_cf, load_image, save_image, estimate_foreground_cf

# Constants
IMAGE_NAME = "pencilcase"
IMAGE_EXTENSION = "jpg"
BLUR_KERNEL_SIZE = 5
THRESH_VALUE_BROAD = 200
THRESH_VALUE_STRICT = 50
THRESH_MAX_VALUE = 255
TRIMAP_KERNEL_SIZE = (9, 9)
N_ITERATIONS = 6
THRESH_BLOCK_SIZE = 11
THRESH_C = 1

# Load image
img = cv2.imread(f"{IMAGE_NAME}.{IMAGE_EXTENSION}", cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# Perform and average to reduce noise
img_cleaned = cv2.medianBlur(img, BLUR_KERNEL_SIZE)

# Create mask broad
_, broad_mask = cv2.threshold(
    img_cleaned, THRESH_VALUE_BROAD, THRESH_MAX_VALUE, cv2.THRESH_BINARY_INV
)
cv2.imwrite(f"./{IMAGE_NAME}_broad_mask.png", broad_mask)

# Create mask strict
_, strict_mask = cv2.threshold(
    img_cleaned, THRESH_VALUE_STRICT, THRESH_MAX_VALUE, cv2.THRESH_BINARY_INV
)
cv2.imwrite(f"./{IMAGE_NAME}_strict_mask.png", strict_mask)

# Generate Trimap
trimap = broad_mask - strict_mask
trimap[trimap == 255] = 153 # setting white pixels to gray
trimap[strict_mask == 255] = 255 # overlapping the strict mask with the black pixels

cv2.imwrite(f"./{IMAGE_NAME}_trimap.png", trimap)

# Alpha Matting
scale = 1.0
temp_img = load_image(f"./{IMAGE_NAME}.{IMAGE_EXTENSION}", "RGB", scale, "box")
temp_trimap = load_image(f"./{IMAGE_NAME}_trimap.png", "GRAY", scale, "nearest")
alpha = estimate_alpha_cf(temp_img, temp_trimap)
save_image(
    f"./{IMAGE_NAME}_alpha.png", alpha
)  # cv2.imwrite doesn't work because it needs to be 0-255

# Foreground and Background
foreground, background = estimate_foreground_cf(temp_img, alpha, return_background=True)
save_image(f"./{IMAGE_NAME}_foreground.png", foreground)
save_image(f"./{IMAGE_NAME}_background.png", background)

# PLOTS
titles = [
    "Original Image",
    f"Mask broad (threshold = {THRESH_VALUE_BROAD})",
    f"Mask strict (threshold = {THRESH_VALUE_STRICT})",
    "Trimap",
    "Alpha",
]
images = [img_cleaned, 
          broad_mask, 
          strict_mask,
          trimap, 
          alpha]

plt.figure(figsize=(30, 20))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], "gray")
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.savefig(f"./{IMAGE_NAME}_results.pdf")
# plt.show()
