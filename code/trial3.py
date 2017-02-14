import cv2
import numpy as np 

image = cv2.imread('1.jpg')
gray  = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

markers = np.zeros_like(gray)
background_pixels = gray[0:100,0:100]

## Some Calculations to cross-check
pixel_mean = np.mean(background_pixels)
pixel_std  = np.std(background_pixels)
pixel_median = np.median(background_pixels)
pixel_min = np.min(background_pixels)
pixel_max = np.max(background_pixels)

'''
bw_lower = int(pixel_median)-int(pixel_std)
bw_upper = int(pixel_median)+int(pixel_std)
'''
color = np.max(image[:50][:50][0]+ image[:50][:50][1] + image[:50][:50][2])
lower = color-10
upper = color+10
#####################################
#print color

mask = np.zeros_like(gray)

for i in range(0,image.shape[0]):
	for j in range(0,image.shape[1]):
		k = image[i][j][0]+ image[i][j][1] + image[i][j][2]
		if (k >=lower and k <= upper) :
			#High possibility that this is background
			mask[i][j] = 0
		else:
			mask[i][j] = 255

cv2.imshow("mask",mask)
cv2.waitKey(0)