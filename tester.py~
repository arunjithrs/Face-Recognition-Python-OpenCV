import cv2
import os
import numpy as np
import faceRecognition as fr
import picam
from tinydb import TinyDB, Query

import time
from picamera.array import PiRGBArray
from picamera import PiCamera

import json, ast


# ------------- THIS IS THE ACTUAL TRAINING PART ------- 

#faces, faceID = fr.labels_for_training_data('trainingImages')

#face_recognizer = fr.train_classifier(faces, faceID)
#face_recognizer.save('trainingData.yml')
#exit()

test_img = cv2.imread('images/item.jpg')
faces_detected, gray_img = fr.faceDetection(test_img)
for (x,y,w,h) in faces_detected:
	cv2.rectangle(test_img, (x,y), (x+w, y+h),(255,0,0), thickness=5)

face_recognizer = cv2.createLBPHFaceRecognizer()
face_recognizer.load('trainingData.yml')

#name = {0: "Arunjith", 1: "Athira", 2: "Rahul", 3: "Muneert", 4: "Nadiya"}

db = TinyDB('users.json')
users = db.all()
name = {}

for user in users:
	for i in user:
		name[int(i)] = user[i]
		
users = ast.literal_eval(json.dumps(name))
new_list = {}

for user in users:
	new_list[int(user)] = users[user]
name = new_list

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(roi_gray)

    print("confidence: ", confidence)
    print("label: ", label)

    fr.draw_rect(test_img, face)
    predicted_name = name[label]

    if confidence > 30:
        continue

    fr.put_text(test_img, predicted_name, x, y)

resized_img = cv2.resize(test_img, (1000, 700))
cv2.imshow("Face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
