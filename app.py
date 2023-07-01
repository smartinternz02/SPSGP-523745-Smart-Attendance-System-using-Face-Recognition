import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
import csv
from flask import Flask, render_template, Response, request
import time

app = Flask(__name__)

path = 'data'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'a') as f:
        now = datetime.now()
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')
        f.writelines(f'{name},{dtString}\n')

encodeListKnown = findEncodings(images)
print('Encoding Complete')


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/take_attendance')
def take_attendance():
    return render_template('take_attendance.html')

@app.route('/start_attendance')
def start_attendance():
    time.sleep(1)  # Add a delay before starting the attendance process
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_attendance')
def stop_attendance():
    time.sleep(1)  # Add a delay before closing the webcam window
    # Release the webcam resources
    cv2.destroyAllWindows()
    return render_template('index.html')

@app.route('/add_person', methods=['GET', 'POST'])
def add_person():
    if request.method == 'POST':
        name = request.form['name']
        image = request.files['image']
        image_path = os.path.join(path, f'{name}.jpg')
        image.save(image_path)
        img = cv2.imread(image_path)
        images.append(img)
        classNames.append(name)
        encodeListKnown.append(findEncodings([img])[0])
        return 'Person added successfully'

    return render_template('add_person.html')


@app.route('/view_attendance')
def view_attendance():
    attendance_data = []
    with open('Attendance.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            attendance_data.append(row)

    # Reverse the attendance_data list to show the recorded attendance first
    attendance_data.reverse()

    return render_template('view_attendance.html', attendance_data=attendance_data)

@app.route('/cancel')
def cancel():
    return 'Program has been canceled'

if __name__ == '__main__':
    app.run()
