import cv2
import numpy as np
def find_crucial_contours(img,contour):
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
	result_b = cv2.bitwise_and(img[:,:,0],img[:,:,0],mask=thresh)
	result_g = cv2.bitwise_and(img[:,:,1],img[:,:,1],mask=thresh)
	result_r = cv2.bitwise_and(img[:,:,2],img[:,:,2],mask=thresh)

	Final_Result = np.zeros_like(img)
	Final_Result[:,:,0] = result_b
	Final_Result[:,:,1] = result_g
	Final_Result[:,:,2] = result_r

	return Final_Result
