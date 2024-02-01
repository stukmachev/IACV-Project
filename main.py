import cv2
import numpy as np
from matplotlib import pyplot as plt

from pymatting import estimate_alpha_cf, load_image, save_image, estimate_foreground_ml

# Constants
IMAGE_NAME = "pencilcase"
IMAGE_EXTENSION = "jpg"
BLUR_KERNEL_SIZE = 5
THRESH_VALUE = 200  # banana 50
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

# Create mask
_, mask = cv2.threshold(
    img_cleaned, THRESH_VALUE, THRESH_MAX_VALUE, cv2.THRESH_BINARY_INV  # banana not inv
)

cv2.imwrite(f"./{IMAGE_NAME}_mask.png", mask)

# Generate Trimap
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, TRIMAP_KERNEL_SIZE)
eroded = cv2.erode(mask, kernel, iterations=N_ITERATIONS)
dilated = cv2.dilate(mask, kernel, iterations=N_ITERATIONS)

trimap = np.full(mask.shape, 128)
trimap[eroded >= 254] = 255
trimap[dilated <= 1] = 0
cv2.imwrite(f"./{IMAGE_NAME}_trimap.png", trimap)

# Alpha Matting
scale = 1.0
temp_img = load_image(f"./{IMAGE_NAME}.{IMAGE_EXTENSION}", "RGB", scale, "box")
# temp_trimap = load_image(f"./{IMAGE_NAME}_trimap.png", "GRAY", scale, "nearest")
temp_trimap = load_image(f"./{IMAGE_NAME}_trimap_manual.png", "GRAY", scale, "nearest") # loading the manual fixed trimap
alpha = estimate_alpha_cf(temp_img, temp_trimap)
save_image(
    f"./{IMAGE_NAME}_alpha.png", alpha
)  # cv2.imwrite doesn't work because it needs to be 0-255

# Foreground and Background
# foreground, background = estimate_foreground_ml(temp_img, alpha, return_background=True)
# save_image(f"./{IMAGE_NAME}_foreground.png", foreground)
# save_image(f"./{IMAGE_NAME}_background.png", background)

# Adaptive Mean Thresholding
adaptive_mean_threshold = cv2.adaptiveThreshold(
    img_cleaned,
    THRESH_MAX_VALUE,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    THRESH_BLOCK_SIZE,
    THRESH_C,
)
cv2.imwrite(f"./{IMAGE_NAME}_mean_adaptive.png", adaptive_mean_threshold)

# PLOTS
titles = [
    "Original Image",
    f"Mask (threshold = {THRESH_VALUE})",
    "Trimap",
    "Adaptive Mean Thresholding",
    "Alpha",
]
images = [img_cleaned, 
          mask, 
        #   trimap, 
          temp_trimap,
          adaptive_mean_threshold, 
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
