import cv2
import face_recognition
import numpy as np
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)


imgDharan = face_recognition.load_image_file('data/Dharan.jpg')
imgDharan = cv2.cvtColor(imgDharan, cv2.COLOR_BGR2RGB)
imgDharan = resize(imgDharan, 0.50)
imgTest = face_recognition.load_image_file('data/Abhinav.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
imgTest = resize(imgTest, 0.50)

faceLoc = face_recognition.face_locations(imgDharan)[0]
encodeDharan = face_recognition.face_encodings(imgDharan)[0]
cv2.rectangle(imgDharan, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 0), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 0), 2)

results = face_recognition.compare_faces([encodeDharan], encodeTest)
faceDis = face_recognition.face_distance([encodeDharan], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

cv2.imshow('Dharan', imgDharan)
cv2.imshow('Dharan Test', imgTest)
cv2.waitKey(0)