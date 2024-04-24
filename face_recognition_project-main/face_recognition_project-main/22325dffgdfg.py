import cv2
import pickle
import numpy as np
import os

# Define absolute paths to model files
model_path = 'C:/Users/austin.reynolds/Downloads/face_recognition_project-main/face_recognition_project-main/data/deploy.prototxt'
weights_path = 'C:/Users/austin.reynolds/Downloads/face_recognition_project-main/face_recognition_project-main/data/res_ssd_300Dim.caffeModel'

# Load pre-trained face detection model
net = cv2.dnn.readNetFromCaffe(model_path, weights_path)

video=cv2.VideoCapture(0)

faces_data=[]

i=0

name=input("Enter Your Name: ")

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Preprocess input image
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)

    # Set input to the model
    net.setInput(blob)

    # Perform inference
    detections = net.forward()

    # Extract faces from detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Adjust confidence threshold as needed
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype('int')
            face = frame[startY:endY, startX:endX]

            # Resize and store face data
            resized_img = cv2.resize(face, (50,50))
            if len(faces_data) <= 100 and i % 10 == 0:
                faces_data.append(resized_img)

            # Draw bounding box
            cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (50,50,255), 1)


    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q') or len(faces_data)==100:
        break

video.release()
cv2.destroyAllWindows()

faces_data=np.asarray(faces_data)
faces_data=faces_data.reshape(100, -1)

if 'names.pkl' not in os.listdir('data/'):
    names=[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names=pickle.load(f)
    names=names+[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
