#__author__ : Shrobon Biswas
'''__Description__ :

This script segments individual herbs from the given images of the dataset.
The extracted herbs are labeled with the first 2 letters of their class name,
and stored in a separate folder named extracted
'''
import numpy as np 
import cv2
import os
from matplotlib import pyplot as plt 
from shrobonutils import find_crucial_contours, make_binary, perform_masking, kmeans
import glob


# The dataset.txt file contains the path of the directories that contain the different classes of herbs 
with open('dataset.txt') as file:
	data = file.read()

data = data.split('\n')	
data = data[:len(data)-1] 
# the path names of the herb classes are now stored in this list called data
# print the data list out to see the the folder names



# Now we open each folder, and extract inidividual herbs from each picture inside that folder
for folder in data:
	print "Folder currently in use %s"%folder
	
	#To grab all the images in a given folder
	imagePaths = glob.glob((folder)+"*.jpg")
	image_counter = 0 # This tracks, how many herbs have beeen extracted till now  


	#This is to extract the class name of the herb
	#The first 2 letter of each class of herb is chosen as the filename followed by the number i 
	name_list = imagePaths[0].split('/')
	category_name = name_list[-2]


	for im in imagePaths:
		img = cv2.imread(im)

		# Since the images have a particular shade of background colour,
		# and since the foreground(herbs) have contrasting colour,
		# K-means will works awesome for this .... as it can easily form two diffferent clusters,
		# based on the very different colour information from the input image.
		label,result = kmeans(img) 


		imgray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		(image, contour , _) =  cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		accepted_contours= find_crucial_contours(img,contour)
		# find_crucial_contours is not a standard opencv function,
		# I defined this function in the helper functions file named shrobonutils.py
		# This is a fullproof method of detecting on the required contours 
		# It eliminates the contours that might have been falsely detected 
		# Go though the function code in shrobonutils.py, to understand the inner workings


		thresh = make_binary(thresh) # i have defined the make_binary_function in shrobonutild.py

		#In this part of code, each herb is
		#1. Cropped ,after performing masking
		#2. 
		for i in range(0,len(accepted_contours)):
			image_counter = image_counter + 1
			cv2.drawContours(thresh, accepted_contours, i, (255,255,255),thickness = -1)
			x,y,w,h = cv2.boundingRect(accepted_contours[i])
			
			#Defining the mask  
			mask = thresh[y:y+h,x:x+w]

			#Erosion has been performed to remove the borders of the herb, which had a tinge of the background color
			kernel = np.ones((5,5),np.uint8)
			erosion = cv2.erode(mask,kernel,iterations = 2)



			cropped_img= img[y:y+h,x:x+w]
			Masked = perform_masking(cropped_img,erosion)
			#We are now saving the extracted herb into the folder, after labelling it (filename)
			cv2.imwrite('/home/shrobon/Assignment2/code/extracted/'+category_name+str(image_counter)+'.jpg',Masked)




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
