import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

image = cv.imread('cube2.png')

# Преобработка изображения
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)


# img = cv.medianBlur(img,5)
th2 = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
titles = ['Original Image', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [blurred, th2, th3]
for i in range(3):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()