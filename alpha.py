import cv2
import numpy as np

# Загрузите изображение
image = cv2.imread('orange.png')
width, height=image.shape[:2]
# Создайте начальную маску
mask = np.zeros(image.shape[:2], np.uint8)

# Задайте прямоугольник, содержащий объект
rect = (0, 0, width, height) # Нужно задать вручную

# Инициализируйте фоновый и передний план модели для GrabCut
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Примените GrabCut
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Преобразование маски GrabCut в двоичное изображение
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
image = image*mask2[:,:,np.newaxis]

# Показать изображение
cv2.imshow('Foreground', image)
cv2.waitKey(0)
cv2.destroyAllWindows()