import numpy as np
import matplotlib.pyplot as plt 
import cv2

from skimage.segmentation import random_walker
from skimage.data import binary_blobs
import skimage

image = cv2.imread('1.jpg')
gray  = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
markers = np.zeros_like(gray)

background_pixels = gray[0:100,0:100]
pixel_mean = np.mean(background_pixels)
pixel_std  = np.std(background_pixels)
pixel_median = np.median(background_pixels)
pixel_min = np.min(background_pixels)
pixel_max = np.max(background_pixels)







'''
print pixel_mean
print pixel_std
print pixel_median
print pixel_min
print pixel_max
'''


for i in range(0,gray.shape[0]):
	for j in range(0,gray.shape[1]):
		if gray[i][j] >= 0 and gray[i][j] <= int(pixel_median) + int(pixel_std):
			markers[i][j] = 0
		else:
			markers[i][j] = 255



labels = random_walker(gray,markers,beta=10,mode='bf')
cv2.imshow("Random Walks ",labels)

#print background_pixels
#cv2.imshow("background chunk",background_pixels)
cv2.waitKey(0)
