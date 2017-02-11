#__author__ : Shrobon Biswas
#__Description__ : To separate out each herb and save it as a single image

import numpy as np 
import cv2
from matplotlib import pyplot as plt 

img = cv2.imread('4.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blurring = cv2.GaussianBlur(imgray,(7,7),0)
ret,thresh = cv2.threshold(blurring,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
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


 

for i in range(0,len(accepted_contours)):
	cv2.drawContours(img, accepted_contours, i , (255,255,255),thickness = -1)

cv2.imshow('Image',img)
cv2.waitKey(0)



