import cv2 
from datetime import datetime
import os
from PIL import Image  # pip install pillow
import numpy as np     # pip install numpy


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
            file_name_path = "data/user." + str(id) + "." + str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Cropped face", face)

        if cv2.waitKey(1) == 13 or int(img_id) == 300:  # Stop on Enter key or after 300 images
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed....")

def train_classifier(data_dir):
    # Filter out non-image files by their extensions
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

    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

train_classifier("data")

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    
    coords = [] 
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        id, pred = clf.predict(gray_img[y:y+h, x:x+w])
        confidence = int(100 * (1 - pred / 300))

        if confidence > 75:
            if id == 1:
                cv2.putText(img, "GURUTEJA", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            elif id == 2:
                cv2.putText(img, "ADIT", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "UNKNOWN", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

def recognize(img, clf, faceCascade):
    coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
    return img

# Loading classifier
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

video_capture = cv2.VideoCapture(0)

while True:
    ret, img = video_capture.read()
    img = recognize(img, clf, faceCascade)
    cv2.imshow("Face Detection", img)

    if cv2.waitKey(1) == 13:  # Stop on Enter key
        break

video_capture.release()
cv2.destroyAllWindows()
