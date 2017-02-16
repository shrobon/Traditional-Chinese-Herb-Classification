#Filename : rgbhistograms.py
#__author__:Shrobon Biswas
#__Description__ : returns the flattened normalized histogram information
import cv2
class RGBHistograms:
	def __init__(self,bins):
		self.bins = bins

	def describe(self,image,mask= None):
		hist = cv2.calcHist([image],[0,1,2],mask,self.bins,[0,256,0,256,0,256])
		cv2.normalize(hist,hist)
		return hist.flatten()

