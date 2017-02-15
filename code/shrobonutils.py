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
		if cont_area >= 10000 and cont_area <= 0.8* image_area:
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
	return Final_Result



def kmeans(image):
    image=cv2.GaussianBlur(image,(7,7),0)
    vectorized=image.reshape(-1,3)
    vectorized=np.float32(vectorized)
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10, 1.0)
    # we have a background and a forebround , so number of segments  should be 2, 
    # that is why i used the parameter as 2
    ret,label,center=cv2.kmeans(vectorized,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    res = center[label.flatten()]
    segmented_image = res.reshape((image.shape))

    return label.reshape((image.shape[0],image.shape[1])),segmented_image.astype(np.uint8)