'''
Project :
------------- 
Face Recognition using Face recognition library


Target Of Project : 
-------------
To record the attendance of students/employees in school/office

Tech Stack:
------------
Python 3.8
OpenCv 4.4.0
numpy 1.18.5
dlib 19.22.99
face-recognition 1.3.0
face-recognition-models 0.3.0
cuda 11.2
cudnn for cuda 11.2


Input: Images
--------
images from dataBase (folder -> image) 

Put Image in Image directory and run command "python databaseGenerator.py"

Process: 
----------
Generate Encoding for the know faces and store them in DataBase.pkl file

dataBase  = {"KnowFaceEncoding":encodings_know,"KnowFaceName":name_know_face}

Output: File
----------
DataBase.pkl

'''


import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import face_recognition
import os
from utils import get_encodings
import pickle as pkl

path: str = "images"
image_list: list = os.listdir(path)

images: list = []
name_know_face: list = []

'''
Iterate through the images dataBase and create a list of images and name of known faces 
'''
for img_name in image_list:
    img: np.ndarray = cv2.imread(f"{path}/{img_name}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)
    name_know_face.append(img_name.split(".")[0])


'''
genetate encoding of images already present in the database 
'''
encodings_know: list = get_encodings(images)

dataBase  = {"KnowFaceEncoding":encodings_know,"KnowFaceName":name_know_face}

with open ("DataBase.pkl",'wb') as file:
    pkl.dump(dataBase,file,protocol=pkl.HIGHEST_PROTOCOL)


print("encodings completed")