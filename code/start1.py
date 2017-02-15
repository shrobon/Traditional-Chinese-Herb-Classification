#__author__ : Shrobon Biswas
'''__Description__ : 
'''
import numpy as np 
import cv2
import os
from matplotlib import pyplot as plt 
from shrobonutils import find_crucial_contours, make_binary, perform_masking, kmeans
import glob


with open('testdataset.txt') as file:
	data = file.read()

data = data.split('\n')	

print data


for folder in data:
	print "Folder currently in use %s"%folder
	#To grab all the images in a given folder
	imagePaths = glob.glob((folder)+"*.jpg")
	# used for numbering and naming the images
	image_counter = 0 

	name_list = imagePaths[0].split('/')
	category_name = name_list[-2]


	for im in imagePaths:
		img = cv2.imread(im)
		label,result = kmeans(img) # k-means worked really well for clustering 



		imgray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		(image, contour , _) =  cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		accepted_contours= find_crucial_contours(img,contour)
		#print "Number of herbs detected in image"
		#print len(accepted_contours)


		thresh = make_binary(thresh)
		for i in range(0,len(accepted_contours)):
			image_counter = image_counter + 1
			cv2.drawContours(thresh, accepted_contours, i, (255,255,255),thickness = -1)
			x,y,w,h = cv2.boundingRect(accepted_contours[i])
			
			#cropping and writing the image 
			mask = thresh[y:y+h,x:x+w]
			cropped_img= img[y:y+h,x:x+w]
			Masked = perform_masking(cropped_img,mask)
			#Masked = preprocess(Masked)
			cv2.imwrite('/home/shrobon/Assignment2/code/testextracted/'+category_name+str(image_counter)+'.jpg',Masked)




'''
for i in range(0,len(accepted_contours)):
	cv2.drawContours(thresh, accepted_contours, i, (255,255,255),thickness = -1)
	
	rect = cv2.minAreaRect(accepted_contours[i])
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	im = cv2.drawContours(thresh,[box],0,(255,255,255),1)
	
	# Strong the bounding box for later use in cropping
	# print box

cv2.imshow("This is it !! ",thresh)
cv2.waitKey(0)
cv2.imshow('Segmented Image',thresh)



Masked = perform_masking(img,thresh)

cv2.imshow("After masking",Masked)
cv2.imwrite('test.jpg',Masked)
cv2.waitKey(0)
'''
