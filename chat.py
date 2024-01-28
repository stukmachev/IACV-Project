import cv2
import numpy as np

# Загрузка исходного цветного изображения
image = cv2.imread('orange.png')

# Преобразование в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение адаптивного порога
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 11, 2)

# Создание цветной маски
color_mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# Применение маски к исходному изображению
result = cv2.bitwise_and(image, color_mask)

# Показать результат
cv2.imshow('Original', image)
cv2.imshow('Threshold', thresh)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()