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


Utilities:
-----------
get_encodings() : Find Encoding if a list of images
put_Attendence() : Put attendance in attendance.csv file   
'''
import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

def get_encodings(images: list) -> list:
    """
    I/p : list of image files \n
    process : generate encoding for the images \n
    O/p : return list of encoding 
    """
    encoding_know_faces = []
    for img in images:
        face_encoding = face_recognition.face_encodings(img)[0]
        encoding_know_faces.append(face_encoding)

    return encoding_know_faces



def put_Attendence(name: str) -> None:
    '''
    Input : name(String) \n

    Process : open the csv file and check if name already exists or not\n
            \t if name already exists continue\n
            \t else record the name and the time of recording\n
    Output: None
    '''
    with open('attendance.csv', 'r+') as file:
        data = file.readlines()
        names_list = []
        for line in data:
            entry = line.split(',')
            names_list.append(entry[0])
        if name not in names_list:
            now = datetime.now()
            now = now.strftime("%H:%M:%S")
            file.writelines(f'\n{name},{now}')


