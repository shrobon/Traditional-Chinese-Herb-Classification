import cv2
import numpy as np
 
class Segment:
    def __init__(self,segments=2):
        #define number of segments, with default 5
        self.segments= segments

    def kmeans(self,image):
        #Preprocessing step
        image=cv2.GaussianBlur(image,(7,7),0)
        vectorized=image.reshape(-1,3)
        vectorized=np.float32(vectorized)
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10, 1.0)
        ret,label,center=cv2.kmeans(vectorized,self.segments,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        res = center[label.flatten()]
    	segmented_image = res.reshape((image.shape))
        return label.reshape((image.shape[0],image.shape[1])),segmented_image.astype(np.uint8)


    def extractComponent(self, image,label_image,label):
       component=np.zeros(image.shape,np.uint8)
       component[label_image==label]=image[label_image==label]
       return component


if __name__== "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True,help = "Path to the image")
    ap.add_argument("-n", "--segments", required = False,type = int,  help = "# of clusters")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])
    if len(sys.argv)==3:
        seg = Segment()
        label,result= seg.kmeans(image)

    else:
		seg = Segment(args["segments"])
		label,result = seg.kmeans(image) 
extracted=seg.extractComponent(image,label,1)

cv2.imshow("input",image)
cv2.imshow("segmented",extracted)
cv2.imshow("result",result)
thresh = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(thresh,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow("OTSU",thresh)



cv2.waitKey(0)
