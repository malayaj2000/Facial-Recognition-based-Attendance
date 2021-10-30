"""
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

Input: Frames
----------
live frame from camara

Notations:
----------------
    camara_number::int           : nth camara in use
    cap::object                  : Video Capture object
    database::File               : DataBase.pkl loaded format
    dist::[]                     : List containing relative distance of each faces
    encodings_know::[]           : List containing Encoding of the known faces
    encodings_curr_frame::[]     : List of encodings of faces in Current scaled frames
    frame::np.ndarray            : Current frame 
    faces_loc_curr_frame::[]     : List of coordinates of faces in Current scaled frames
    indx::int                    : Index of face 
    match::bool[]                : Boolean list of matched faces in Current scaled frames 
    name_know_face::[]           : List containing name of the known faces
    ret::bool                    : Bool value is true as long as camara is open else false
    small_frame::np.ndarray      : Scaled frame to 0.25 * original frame size



Process: 
------------
    1 .Opening DataBase.pkl 
    2 . Open Camara and get Frames
        2.1 frames scaled/resized  to 1/4 th of its original size (for easy processing)
        2.2 frames converted to RGB image format
    3 . Find Location of faces and their coressponding encodings
    4. Loop through KnownEncoding and dispaly it and mark its attendance  

Output: 
----------
Csv(comma separated values) file containing name of the preson and time stamp of attendance  
"""

import pickle as pkl
from utils import put_Attendence
from datetime import datetime
import os
import face_recognition
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

try:
    '''
    1 .Opening DataBase.pkl 
    '''

    with open("Database.pkl", 'rb') as file:
        database = pkl.load(file)
        encodings_know: list = database["KnowFaceEncoding"]
        name_know_face: list = database["KnowFaceName"]
        print("Database successfully loaded")

        camara_number: int = 1
        '''
            2 . Open Camara and get Frames
                2.1 frames scaled/resized  to 1/4 th of its original size (for easy processing)
                2.2 frames converted to RGB image format
            3 . Find Location of faces and their coressponding encodings
            4. Loop through KnownEncoding and dispaly it and mark its attendance      
        '''
        cap = cv2.VideoCapture(camara_number)
        while cap.isOpened():
            ret: bool
            frame: np.ndarray
            ret, frame = cap.read()  # get Frames
            small_frame: np.ndarray = cv2.resize(
                frame, (0, 0), None, fx=0.25, fy=0.25)  # frames scaled (2.1)
            # frames converted to RGB(2.2)
            small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            faces_loc_curr_frame: list = face_recognition.face_locations(
                small_frame)   # Find Location of faces
            # faces coressponding encodings
            encodings_curr_frame: list = face_recognition.face_encodings(
                small_frame, faces_loc_curr_frame)
            # Loop through KnownEncoding
            for encodings_curr, face_loc_curr in zip(encodings_curr_frame, faces_loc_curr_frame):
                match: list = face_recognition.compare_faces(
                    encodings_know, encodings_curr)
                dist: np.ndarray = face_recognition.face_distance(
                    encodings_know, encodings_curr)

                indx: int = np.argmin(dist)
                if match[indx]:
                    # dispaly it and mark its attendance if face found
                    put_Attendence(name_know_face[indx].upper())
                    cv2.rectangle(frame, (face_loc_curr[3] * 4, face_loc_curr[0] * 4),
                                  (face_loc_curr[1] * 4, face_loc_curr[2] * 4), (255, 255, 0), 2)
                    cv2.rectangle(frame, (face_loc_curr[3] * 4, face_loc_curr[2] * 4-25),
                                  (face_loc_curr[1] * 4, face_loc_curr[2] * 4), (255, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name_know_face[indx].upper(), ((face_loc_curr[3] * 4, face_loc_curr[2] * 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    # dispaly "Unknown Face" if face not found
                    cv2.rectangle(frame, (face_loc_curr[3] * 4, face_loc_curr[0] * 4),
                                  (face_loc_curr[1] * 4, face_loc_curr[2] * 4), (0, 0, 255), 2)
                    cv2.rectangle(frame, (face_loc_curr[3] * 4, face_loc_curr[2] * 4-25),
                                  (face_loc_curr[1] * 4, face_loc_curr[2] * 4), (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, "Unknown Face", ((face_loc_curr[3] * 4, face_loc_curr[2] * 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("frame", frame)
            if cv2.waitKey(10) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

except:
    raise Exception("Could not open database.pkl")
