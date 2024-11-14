# FaceTrace
 FaceTrace, an automated attendance system, uses facial recognition to capture real-time images from classroom CCTV. It identifies students, logs attendance, and securely stores data. Contactless tracking, reduced human error, and enhanced efficiency are achieved. Privacy is ensured with encrypted data storage. Scalability adapts to various class sizes.

How this code Works
1. Set Up the Environment
    * Ensure you have Python, OpenCV, PIL (Pillow), and NumPy installed.
    * Download the haarcascade_frontalface_default.xml file and place it in the same directory as the code.
2. Run the Application
    * Run the script to open the Tkinter GUI window.
3. Generate Dataset
    * Enter Name, Roll No, and Class in the respective fields.
    * Click "Generate Dataset" to capture 200 grayscale images of the user’s face, which are stored in the datadirectory.
      
      <img width="400" alt="image" src="https://github.com/user-attachments/assets/c8b12b17-83d6-4079-a023-bb45d4391729">
      Capturing images of individuals to train the recognition system.

4. Train the Classifier
    * Click "Train Classifier" to train the LBPH face recognizer model using the dataset created.
      
       <img width="292" alt="image" src="https://github.com/user-attachments/assets/103acbfa-96de-4a5a-9391-6fefe23ce9e4">
      Determining identities by recognizing faces and associating them with names and IDs after retrieving similar faces from a database.
      
    * The trained model is saved as classifier.xml for real-time recognition.

5. Take Attendance
    * Click "GET Attendance" to start the real-time face detection and attendance logging.
    * Recognized faces are labeled, and attendance data (Name, Date, and Time) is saved in a CSV file (<current_date>_attendance.csv).
  
       <img width="212" alt="image" src="https://github.com/user-attachments/assets/fcf8f079-6fa8-48b5-8b8b-4c0b862ce8d6">
      Recording attendance exclusively for faces that are recognized from the database and disregarding any unknown faces.
      
    * Press Enter to stop the attendance recording.
6. Review Attendance Data
    * The attendance CSV file can be found in the project directory with entries for each recognized user.
  
      <img width="239" alt="image" src="https://github.com/user-attachments/assets/e78e11df-7e7a-49ad-83ba-61336b21ea00">
      This is the format of the attendance sheet output in Excel, which includes names, dates, and times.


THANK YOU
