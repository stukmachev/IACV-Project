import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_NAME = "basketball"
IMAGE_EXTENSION = "png"

# Image download
image_path = f"{IMAGE_NAME}.{IMAGE_EXTENSION}"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Transition into shades of gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gauss Blur implementation
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Sobel operator implementation for axises X and Y
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

# Gradient calculation
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

# Gradient normalization
gradient_magnitude = np.uint8(gradient_magnitude / gradient_magnitude.max() * 255)

# Invert colors for better visualization 
inverted_edges = cv2.bitwise_not(gradient_magnitude)

equalized = cv2.equalizeHist(inverted_edges)

# Saving and output
cv2.imwrite(f"./{IMAGE_NAME}_gradient.png", equalized)

plt.figure(figsize=(20, 5))
plt.subplot(131), plt.imshow(gray, cmap='gray'), plt.title('Original Gray'), plt.axis('off')
plt.subplot(133), plt.imshow(inverted_edges, cmap='gray'), plt.title('Gradient Magnitude Inverted'), plt.axis('off')
plt.subplot(132), plt.imshow(equalized, cmap='gray'), plt.title('Equalized Gradient'), plt.axis('off')
plt.show()