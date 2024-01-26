import argparse
import time
import cv2
print("hiii")

image = cv2.imread(r'dog.jpg') 
cv2.imshow('image', image) 
  
# waits for user to press any key 
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows() 
