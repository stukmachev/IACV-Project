import cv2
import numpy as np
import os

# Загрузка изображения
filename='fan.png'
image = cv2.imread(filename)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Создание простого trimap
# Предполагаем, что белые области - это передний план, черные - фон, серый - неопределенные области
threshold = 128  # Это значение может варьироваться в зависимости от вашего изображения
foreground = (gray > threshold).astype(np.uint8) * 255
background = (gray < threshold).astype(np.uint8) * 255
uncertain = np.uint8((foreground == 0) & (background == 0)) * 128

trimap = foreground + uncertain

base_name = os.path.splitext(filename)[0]
trimap_file_name = f"{base_name}_trimap.png"

# Сохранение файла
cv2.imwrite(trimap_file_name, trimap)

# Показать результат
cv2.imshow('Trimap', trimap)
cv2.waitKey(0)
cv2.destroyAllWindows()