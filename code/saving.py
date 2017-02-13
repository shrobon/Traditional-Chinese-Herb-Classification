#__author__ : Shrobon Biswas
'''__Description__ : 
1. To detect correct number of contours
2. Performing masking to segment out the image 

'''

import numpy as np 
import cv2
from matplotlib import pyplot as plt 

img = cv2.imread('4.jpg')
#cv2.imshow("Original Image",img)

imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#blurring = cv2.GaussianBlur(imgray,(7,7),0)
ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
thresh = cv2.medianBlur(thresh,11)
(image, contour , _) =  cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

image_area = img.shape[0] * img.shape[1]
arealist = []
accepted_contours =[]

for i in contour:
	cont_area = cv2.contourArea(i)
	if cont_area >= 10000 and cont_area <= 0.8 * image_area:
		arealist.append(cont_area)
		accepted_contours.append(i)

print "Number of herbs detected in image"
print len(accepted_contours)


for i in range(0,thresh.shape[0]):
	for j in range(0,thresh.shape[1]):
		if thresh[i][j] < 50:
			thresh[i][j] = 0
		else:
			thresh[i][j] = 255



for i in range(0,len(accepted_contours)):
	cv2.drawContours(thresh, accepted_contours, i, (255,255,255),thickness = -1)

	rect = cv2.minAreaRect(accepted_contours[i])
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	im = cv2.drawContours(thresh,[box],0,(255,255,255),1)

	#Strong the bounding box for later use in cropping
	print rect
cv2.imshow('Segmented Image',thresh)





 


result_b = cv2.bitwise_and(img[:,:,0],img[:,:,0],mask=thresh)
result_g = cv2.bitwise_and(img[:,:,1],img[:,:,1],mask=thresh)
result_r = cv2.bitwise_and(img[:,:,2],img[:,:,2],mask=thresh)

Final_Result = np.zeros_like(img)
Final_Result[:,:,0] = result_b
Final_Result[:,:,1] = result_g
Final_Result[:,:,2] = result_r

cv2.imshow("After masking",Final_Result)
cv2.waitKey(0)

