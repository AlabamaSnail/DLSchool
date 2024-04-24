import cv2
import numpy as np
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import pickle
from win32com.client import Dispatch

# Function to speak
def speak(text):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(text)

# Define absolute paths to model files
model_path = 'C:/Users/austin.reynolds/Downloads/face_recognition_project-main/face_recognition_project-main/data/deploy.prototxt'
weights_path = 'C:/Users/austin.reynolds/Downloads/face_recognition_project-main/face_recognition_project-main/data/res_ssd_300Dim.caffeModel'

# Load pre-trained face detection model
net = cv2.dnn.readNetFromCaffe(model_path, weights_path)

# Load saved face recognition model and data
with open('C:/Users/austin.reynolds/Downloads/face_recognition_project-main/face_recognition_project-main/data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('C:/Users/austin.reynolds/Downloads/face_recognition_project-main/face_recognition_project-main/data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(FACES, LABELS)

# Initialize video capture
video = cv2.VideoCapture(0)

# Define confidence threshold
confidence_threshold = 0.5  # Adjust as needed

# Define column names for attendance CSV
COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Preprocess input image
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)

    # Set input to the model
    net.setInput(blob)

    # Perform inference
    detections = net.forward()

    # Post-process detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype('int')
            face = frame[startY:endY, startX:endX]

            # Perform face recognition
            # Perform face recognition
            resized_img = cv2.resize(face, (50, 50)).flatten().reshape(1, -1)

            # Get the predicted label and its probability
            output_label = knn.predict(resized_img)
            confidence = knn.predict_proba(resized_img).max()

            # Check if the confidence is above the threshold and if the label is valid
            if confidence > confidence_threshold and output_label[0] in LABELS:
                # Draw bounding box and display recognized label
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, str(output_label[0]), (startX, startY - 15), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (255, 255, 255), 2)
                cv2.putText(frame, str(confidence), (startX, startY - 55), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            else:
                # Skip drawing bounding box for unrecognized faces
                # Display "Unknown Face" if face is not recognized
                cv2.putText(frame, "Unknown Face", (startX, startY - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                            2)
                cv2.putText(frame, str(confidence), (startX, startY - 55), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

            # Save attendance
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
            attendance = [str(output_label[0]), str(timestamp)]
            exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

            if exist:
                with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance)
            else:
                with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COL_NAMES)
                    writer.writerow(attendance)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check for key press
    key = cv2.waitKey(1)
    if key == ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)
    elif key == ord('q'):
        break

# Release video capture and close all windows
video.release()
cv2.destroyAllWindows()
