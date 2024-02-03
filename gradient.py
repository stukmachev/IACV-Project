import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_NAME = "fan1"
IMAGE_EXTENSION = "jpg"

# Загрузка изображения
image_path = 'fan1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Преобразование в градации серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение Гауссова размытия
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Применение оператора Собеля для вычисления градиентов по оси X и Y
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

# Вычисление величины градиента
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

# Нормализация величины градиента для отображения
gradient_magnitude = np.uint8(gradient_magnitude / gradient_magnitude.max() * 255)


inverted_edges = cv2.bitwise_not(gradient_magnitude)

th3 = cv2.adaptiveThreshold(inverted_edges,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

equalized = cv2.equalizeHist(inverted_edges)


laplacian = cv2.Laplacian(th3,cv2.CV_64F)


cv2.imwrite(f"./{IMAGE_NAME}_gradient.png", th3)

# Создаем ядро для дилатации
# kernel = np.ones((3,3), np.uint8)

# # Применяем дилатацию
# dilated_edges = cv2.dilate(inverted_edges, kernel, iterations=2)
# Визуализация величины градиента
plt.figure(figsize=(20, 5))
plt.subplot(231), plt.imshow(gray, cmap='gray'), plt.title('Original Gray')
plt.subplot(232), plt.imshow(inverted_edges, cmap='gray'), plt.title('Gradient Magnitude Inverted (Sobel)')
plt.subplot(233), plt.imshow(equalized, cmap='gray'), plt.title('thresholding')
plt.subplot(234), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X')
plt.subplot(235), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y')
plt.subplot(236), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian')
plt.axis('off')  # Скрыть оси для лучшей визуализации
plt.show()