from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
with open('data/names.pkl', 'rb') as w:
    LABELS=pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

knn=KNeighborsClassifier(n_neighbors=25)
knn.fit(FACES, LABELS)

imgBackground=cv2.imread("background.png")

COL_NAMES = ['NAME', 'TIME']

while True:
    ret,frame=video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    #faces=facedetect.detectMultiScale(gray, 1.3 ,5)

    average_brightness = np.mean(gray)
    confidence_threshold = 0.95

    if average_brightness < 100:  # Adjust this threshold based on your observations
        confidence_threshold = 0.85
    elif average_brightness > 200:  # Adjust this threshold based on your observations
        confidence_threshold = 0.98


    for (x,y,w,h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output_label = knn.predict(resized_img)
        confidence = knn.predict_proba(resized_img).max()  # Get the maximum confidence score
        #print(confidence)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        if confidence < confidence_threshold:
            cv2.putText(frame, "Unknown Student", (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, str(confidence), (x, y - 55), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(frame, str(output_label[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x, y - 55), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            #speak("Detecting" + str(output_label[0]))
            #speak(RandomFact)

        attendance = [str(output_label[0]), str(timestamp)]
        # Calculate the position to place the camera feed in the middle of the background
    bg_height, bg_width, _ = imgBackground.shape
    cam_height, cam_width, _ = frame.shape

    x_offset = (bg_width - cam_width) // 2
    y_offset = 50

    imgBackground[y_offset:y_offset + cam_height, x_offset:x_offset + cam_width] = frame

    cv2.imshow("Frame", imgBackground)
    k=cv2.waitKey(1)
    if k==ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()

