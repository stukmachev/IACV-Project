import cv2
import numpy as np


im = cv2.imread('dog.png')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if cv2.contourArea(contours[i]) < 100:#remove small contour areas
        continue
    cv2.fillPoly(thresh, pts=[contours[i]], color=(255))

cv2.drawContours(im, contours, -1, (0,255,0), 3)
cv2.imshow("title", im)
cv2.waitKey(0)