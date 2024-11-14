import cv2
from datetime import datetime
import os
from PIL import Image
import numpy as np
import csv
import tkinter as tk
from tkinter import messagebox

# Initialize Tkinter window
window = tk.Tk()
window.title("Face Recognition System")

# Set up labels and input fields for name, roll no, and class
l1 = tk.Label(window, text="NAME", font=("Algerian", 20))
l1.grid(column=0, row=0)
t1 = tk.Entry(window, width=50, bd=5)
t1.grid(column=1, row=0)

l2 = tk.Label(window, text="ROLL NO", font=("Algerian", 20))
l2.grid(column=0, row=1)
t2 = tk.Entry(window, width=50, bd=5)
t2.grid(column=1, row=1)

l3 = tk.Label(window, text="CLASS", font=("Algerian", 20))
l3.grid(column=0, row=2)
t3 = tk.Entry(window, width=50, bd=5)
t3.grid(column=1, row=2)

def generate_dataset():
    if t1.get() == "" or t2.get() == "" or t3.get() == "":
        messagebox.showinfo('Result', 'Please provide complete details of the user')
    else:
        name = t1.get()
        person_id = t2.get()  # Uses the ROLL NO as the ID
        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            return img[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]] if len(faces) > 0 else None
        
        cap = cv2.VideoCapture(0)
        img_id = 0
        
        while True:
            ret, frame = cap.read()
            cropped_face = face_cropped(frame)
            if cropped_face is not None:
                img_id += 1
                face = cv2.resize(cropped_face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = f"data/user.{person_id}.{img_id}.jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Cropped face", face)
                
            if cv2.waitKey(1) == 13 or img_id == 200:  # 200 images for training
                break
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Dataset generation completed!')

def train_classifier():
    data_dir = "data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jpg")]
    faces, ids = [], []
    
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
        faces.append(imageNp)
        ids.append(id)
        
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, np.array(ids))
    clf.write("classifier.xml")
    messagebox.showinfo('Result', 'Training dataset completed')

def get_attendance():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf, detected_names, writer):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)

        for (x, y, w, h) in faces:
            id, pred = clf.predict(gray_img[y:y+h, x:x+w])
            confidence = int(100 * (1 - pred / 300))
            
            if confidence > 80: # Change the Confidence level Accordingly
                name = "UNKNOWN"
                if id == 1:
                    name = "USER1"
                elif id == 2:
                    name = "USER2"
                
                # Check if the name has not been added before, then record it
                if name != "UNKNOWN" and name not in detected_names:
                    detected_names.add(name)  # Add to set to avoid duplicates
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    writer.writerow([len(detected_names), name, current_date, current_time])
                
                cv2.putText(img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for unknown faces
                cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

    # Load the face cascade and classifier model
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    detected_names = set()  # Set to track detected names and avoid duplicates

    # Open CSV file to record attendance
    with open(f"{current_date}_attendance.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["S.NO", "Name", "Date", "Time"])  # Write header
        
        video_capture = cv2.VideoCapture(0)
        
        while True:
            ret, img = video_capture.read()
            draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), clf, detected_names, writer)
            cv2.imshow("Face Detection", img)
            
            if cv2.waitKey(1) == 13:  # Press 'Enter' to stop
                break
                
        video_capture.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Attendance has been recorded!')

# Set up buttons in Tkinter window
b1 = tk.Button(window, text="Generate Dataset", font=("Algerian", 20), bg="pink", fg="black", command=generate_dataset)
b1.grid(column=2, row=4)

b2 = tk.Button(window, text="Train Classifier", font=("Algerian", 20), bg="orange", fg="red", command=train_classifier)
b2.grid(column=0, row=4)

b3 = tk.Button(window, text="GET Attendance", font=("Algerian", 20), bg="green", fg="orange", command=get_attendance)
b3.grid(column=1, row=4)

window.geometry("1080x720")
window.mainloop()
