import cv2
from datetime import datetime
import os
from PIL import Image
import numpy as np
import csv

def generate_dataset(): 
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        if faces == ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
        return cropped_face
    
    cap = cv2.VideoCapture(0)
    id = 1
    img_id = 0
    
    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id += 1
            face = cv2.resize(face_cropped(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = f"data/user.{id}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped face", face)
             
        if cv2.waitKey(1) == 13 or int(img_id) == 300:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed...")

def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jpg")]
    faces = []
    ids = []
     
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
        faces.append(imageNp)
        ids.append(id)
         
    ids = np.array(ids)
     
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

train_classifier("data")

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf, detected_names):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        id, pred = clf.predict(gray_img[y:y+h, x:x+w])
        confidence = int(100 * (1 - pred / 300))
         
        if confidence > 75:
            name = "UNKNOWN"
            if id == 1:
                name = "GURUTEJA "
            elif id == 2:
                name = "ADIT"
            elif id == 3:
                name = "LOKESH"
            
            if name != "UNKNOWN" and name not in detected_names:
                detected_names.add(name)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                writer.writerow([len(detected_names), name, current_date, current_time])
            
            cv2.putText(img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

def recognize(img, clf, faceCascade, detected_names):
    coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf, detected_names)
    return img

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

with open(current_date + '.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["S.NO", "Name", "Date", "Time"])

    video_capture = cv2.VideoCapture(0)
    detected_names = set()

    while True:
        ret, img = video_capture.read()
        img = recognize(img, clf, faceCascade, detected_names)
        cv2.imshow("Face Detection", img)
         
        if cv2.waitKey(1) == 13:
            break

    video_capture.release()
    cv2.destroyAllWindows()
