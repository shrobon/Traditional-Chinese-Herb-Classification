import cv2
import numpy as np


def find_crucial_contours(img,contour):
	##################################################
	# This function helps to remove the false contours
	# and returns only the contours needed by our application
	# Change the threshold value, and ratio of the image area (0.8 in my case )
	# and size of the contour you are looking at (above 10000 in my case)
	# as per your application
	##################################################
	arealist=[]
	accepted_contours=[]
	image_area = img.shape[0] * img.shape[1]
	for i in contour:
		cont_area = cv2.contourArea(i)
		if cont_area >= 10000 and cont_area <= 0.8 * image_area:
			arealist.append(cont_area)
			accepted_contours.append(i)

	return accepted_contours





def make_binary(thresh):
	for i in range(0,thresh.shape[0]):
		for j in range(0,thresh.shape[1]):
			if thresh[i][j] < 50:
				thresh[i][j] = 0
			else:
				thresh[i][j] = 255

	return thresh


def perform_masking(img,thresh):
	# Performs masking of 2 images using bitwise_and 
	# to extract the region of interst
	
	result_b = cv2.bitwise_and(img[:,:,0],img[:,:,0],mask=thresh)
	result_g = cv2.bitwise_and(img[:,:,1],img[:,:,1],mask=thresh)
	result_r = cv2.bitwise_and(img[:,:,2],img[:,:,2],mask=thresh)

	Final_Result = np.zeros_like(img)
	Final_Result[:,:,0] = result_b
	Final_Result[:,:,1] = result_g
	Final_Result[:,:,2] = result_r
	'''
	thresh1 = cv2.bitwise_not(thresh)
	result_b1 = cv2.bitwise_or(Final_Result[:,:,0],Final_Result[:,:,0],mask=thresh1)
	result_g1 = cv2.bitwise_or(Final_Result[:,:,0],Final_Result[:,:,0],mask=thresh1)
	result_r1 = cv2.bitwise_or(Final_Result[:,:,0],Final_Result[:,:,0],mask=thresh1)

	Final_Result[:,:,0] = result_b1
	Final_Result[:,:,1] = result_g1
	Final_Result[:,:,2] = result_r1
	'''

	return Final_Result

def preprocess(image):
	for i in range(0,image.shape[0]):
		for j in range(0,image.shape[1]):
			if image[i][j][:3] < 1:
				image[i][j][:3] = 255

	return image
