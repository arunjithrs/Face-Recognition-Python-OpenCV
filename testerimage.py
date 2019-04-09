import cv2
import os
import numpy as np
import faceRecognition as fr

cap = cv2.VideoCapture(0)
while(True):
	ret,frame = cap.read()
	cv2.imshow('img1', frame)
	if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
		cv2.imwrite('images/c1.png',frame)
		cv2.destroyAllWindows()
		break

cap.release()

test_img = cv2.imread('images/c1.png')
faces_detected, gray_img = fr.faceDetection(test_img)
print("Face detected:", faces_detected)

# for (x,y,w,h) in faces_detected:
# 	cv2.rectangle(test_img, (x,y), (x+w, y+h),(255,0,0), thickness=5)



# faces, faceID = fr.labels_for_training_data('trainingImages')
# face_recognizer = fr.train_classifier(faces, faceID)
# face_recognizer.save('trainingData.yml')
# exit()


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')

exit();

name = {0: "Arunjith", 1: "Athira", 2: "Rahul", 3: "Muneert", 4: "Nadiya", 5: "Aiswarya"}

for face in faces_detected:
	(x,y,w,h) = face
	roi_gray = gray_img[y:y+h, x:x+h]

	label, confidence = face_recognizer.predict(roi_gray)

	print("confidence: ", confidence)
	print("label: ", label)

	fr.draw_rect(test_img, face)
	predicted_name = name[label]
	isIdentified = False
	if confidence > 30:
		continue
	else
		isIdentified = True
	
	print(predicted_name)
	return isIdentified

	# fr.put_text(test_img, predicted_name, x, y)

resized_img = cv2.resize(test_img, (1000, 700))
cv2.imshow("Face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
