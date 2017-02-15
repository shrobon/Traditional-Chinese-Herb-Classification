from rgbhistograms import RGBHistograms
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2
import glob 
import numpy as np 

path_to_dataset = '/home/shrobon/Assignment2/code/extracted/'
imagePaths = sorted(glob.glob((path_to_dataset)+'*.jpg'))
#print imagePaths
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


#Getting all the class names i have in my dataset
targetNames = np.unique(target)
le = LabelEncoder() #This will required to encode the class names
target = le.fit_transform(target)

(trainData,testData,trainTarget,testTarget) = train_test_split(data,target,test_size = 0.3, random_state=42)

model = RandomForestClassifier(n_estimators = 25, random_state=84)
model.fit(trainData,trainTarget)

print(classification_report(testTarget,model.predict(testData),target_names=targetNames))


