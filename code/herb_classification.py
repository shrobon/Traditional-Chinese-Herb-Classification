#__Filename__ : herb_classification.py
#__author__ : Shrobon Biswas
'''__Description__ :

1. This script first read the extracted images from the folder.
2. The feature-vector of this image is then determined
3. 70% of images from the extracted folder are used for training
4. Training is done using a simple Random Forest Classifier
5. I have used 25 trees for random forest classifier. You can calibrate this value further.
6. After the training, i have saved the model, so that it can be used directly, later.
7. I have stored one herb from each herb class in the folder named testClassificationImages
8. I use the images mentioned in step 7, to test how well my classifier performs.

Verdict: I have conducted a few experiments and it mostly gets the classification right. 
'''
from rgbhistograms import RGBHistograms
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2
import glob 
import numpy as np 
import warnings
import cPickle 
warnings.filterwarnings("ignore")


#This folder contains the extracted herb images
path_to_dataset = '/home/shrobon/Assignment2/code/extracted/'
imagePaths = sorted(glob.glob((path_to_dataset)+'*.jpg'))


data = [] #  I will keep the image info here
target = []# This will contain the labels


desc = RGBHistograms([8,8,8])



#iterating over each image 
for i in imagePaths:
	image = cv2.imread(i)
	features = desc.describe(image) # a flattened histogram will be returned

	#updating the feature vector
	data.append(features)
	
	#updating the corresponding labels 
	label = i.split('/')
	label = label[len(label)-1][:2]
	target.append(label)
	#print label


#Getting all the unique class names i have in my dataset
targetNames = np.unique(target)
le = LabelEncoder() #This will required to encode the class names as 0's and 1's
target = le.fit_transform(target)
print target
(trainData,testData,trainTarget,testTarget) = train_test_split(data,target,test_size = 0.3, random_state=42)

model = RandomForestClassifier(n_estimators = 25, random_state=84)
model.fit(trainData,trainTarget)
print classification_report(testTarget,model.predict(testData),target_names=targetNames)



#Saving my classifier to disk, so that i can use it later just for prediction
f = open("model.cpickle","wb")
f.write(cPickle.dumps(model))
f.close()




#I will now test my classification to see how good my classifier works 
TestPaths = '/home/shrobon/Assignment2/code/testClassificationImages/'
TestImages = sorted(glob.glob((TestPaths)+'*.jpg'))


print "Testing the performance of my classifier"
print "::::::::::::::::::::::::::::::::::::::::"
test_counter = 0
for i in range(0,len(TestImages)):
	test_counter = test_counter+1

	image = cv2.imread(TestImages[i])
	feature = desc.describe(image)
	flower = le.inverse_transform(model.predict(feature))[0]
	


	x = TestImages[i].split('/')
	x = x[len(x)-1][:2]
	print "Test number    :->{}".format(test_counter)
	print "Displayed herb :-> {}".format(x.upper()) 
	print "Predicted herb :-> {}".format(flower.upper())
	if flower == x :
		print "VERDICT : Correct !!! "
	else:
		print "VERDICT : Oops !! i missed it :( sorry "
	print ":::::::::::::::::::::::::::::::::::::::::::"
	cv2.imshow("The herb",image)
	cv2.waitKey(0)