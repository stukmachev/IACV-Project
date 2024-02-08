import cv2
import numpy as np
from matplotlib import pyplot as plt

from pymatting import estimate_alpha_cf, load_image, save_image, estimate_foreground_cf

# Constants
IMAGE_NAME = "pencilcase"
IMAGE_EXTENSION = "jpg"
THRESH_BINARY = cv2.THRESH_BINARY
THRESH_VALUE_BROAD = 200
THRESH_VALUE_STRICT = 50
THRESH_MAX_VALUE = 255

# Load image
img = cv2.imread(f"{IMAGE_NAME}.{IMAGE_EXTENSION}", cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# Perform and average to reduce noise
img_cleaned = cv2.medianBlur(img, 5) # BLUR_KERNEL_SIZE = 5

# Create broad mask
_, broad_mask = cv2.threshold(
    img_cleaned, THRESH_VALUE_BROAD, THRESH_MAX_VALUE, THRESH_BINARY
)
cv2.imwrite(f"./{IMAGE_NAME}_broad_mask.png", broad_mask)

# Create strict mask
_, strict_mask = cv2.threshold(
    img_cleaned, THRESH_VALUE_STRICT, THRESH_MAX_VALUE, THRESH_BINARY
)
cv2.imwrite(f"./{IMAGE_NAME}_strict_mask.png", strict_mask)

# Generate Trimap
if (THRESH_BINARY == cv2.THRESH_BINARY):
    trimap = strict_mask - broad_mask
    trimap[trimap == 255] = 153 # setting white pixels to gray
    trimap[broad_mask == 255] = 255 # overlapping the strict mask with the black pixels
else:
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
