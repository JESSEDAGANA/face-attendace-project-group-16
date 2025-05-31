# Import required libraries for image processing, file handling, web framework, and machine learning
import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import shutil

# Initialize Flask application
app = Flask(__name__)

# Define constant for number of images to capture for a new user
nimgs = 10

# Define the AttendanceSystem class to encapsulate functionality
class AttendanceSystem:
    def __init__(self):
        # Initialize instance attributes that were previously global variables
        self.imgBackground = self.load_background()  # Load background image
        self.datetoday = date.today().strftime("%m_%d_%y")  # Get current date for file naming
        self.datetoday2 = date.today().strftime("%d-%B-%Y")  # Get current date for display
        # Load Haar Cascade classifier for face detection
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Check if the classifier loaded successfully
        if self.face_detector.empty():
            raise FileNotFoundError("Could not load haarcascade_frontalface_default.xml")
        self.csv_path = f'Attendance/Attendance-{self.datetoday}.csv'  # Path to attendance CSV
        
        # Create necessary directories if they don't exist
        for folder in ['Attendance', 'static', 'static/faces']:
            if not os.path.isdir(folder):
                os.makedirs(folder)
        
        # Initialize attendance CSV file for the current date if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:  # Fixed typo: 'asAND' to 'as f'
                # Write header for CSV file
                f.write('Name,Roll,Time')

    # Function to load and validate background image
    def load_background(self):
        # Attempt to load background.png
        try:
            img = cv2.imread("background.png")
            # Check if image failed to load
            if img is None:
                print("Error: background.png not found or invalid. Using fallback.")
                # Return a black 1280x720 image as fallback
                return np.zeros((720, 1280, 3), dtype=np.uint8)
            # Check if image dimensions are too small
            if img.shape[0] < 720 or img.shape[1] < 1280:
                print(f"Error: background.png too small {img.shape}. Using fallback.")
                return np.zeros((720, 1280, 3), dtype=np.uint8)
            # Resize image if dimensions are larger than 1280x720
            if img.shape[0] > 720 or img.shape[1] > 1280:
                print(f"Resizing background.png from {img.shape} to (720, 1280, 3)")
                img = cv2.resize(img, (1280, 720))
            print("Background loaded successfully. Shape:", img.shape)
            return img
        # Handle any errors during loading and return fallback image
        except Exception as e:
            print(f"Error loading background.png: {e}. Using fallback.")
            return np.zeros((720, 1280, 3), dtype=np.uint8)

    # Function to count total registered users
    def totalreg(self):
        # Return the number of user folders in static/faces
        return len(os.listdir('static/faces'))

    # Function to detect faces in an image
    def extract_faces(self, img):
        try:
            # Convert image to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect faces with default parameters
            face_points = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            # If no faces detected, try with more lenient parameters
            if len(face_points) == 0:
                face_points = self.face_detector.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=1, minSize=(10, 10))
            return face_points
        # Handle errors during face detection and return empty list
        except Exception as e:
            print(f"Error extracting faces: {e}")
            return []

    # Function to identify a face using a trained model
    def identify_face(self, facearray):
        try:
            # Load the pre-trained KNeighborsClassifier model
            model = joblib.load('static/face_recognition_model.pkl')
            # Predict the identity of the face
            return model.predict(facearray)
        # Handle errors during prediction and return "Unknown" identifier
        except Exception as e:
            print(f"Error identifying face: {e}")
            return ["Unknown_0"]

    # Function to train a face recognition model
    def train_model(self):
        # Initialize lists to store face data and labels
        faces = []
        labels = []
        # Get list of registered users
        userlist = os.listdir('static/faces')
        # Iterate through each user's folder
        for user in userlist:
            # Iterate through each image in the user's folder
            for imgname in os.listdir(f'static/faces/{user}'):
                # Read the image
                img = cv2.imread(f'static/faces/{user}/{imgname}')
                # Resize image to 50x50 pixels for consistency
                resized_face = cv2.resize(img, (50, 50))
                # Flatten image and add to faces list
                faces.append(resized_face.ravel())
                # Add user label to labels list
                labels.append(user)
        # Train model only if faces are available
        if faces:
            # Convert faces list to numpy array
            faces = np.array(faces)
            # Initialize KNeighborsClassifier with 5 neighbors
            knn = KNeighborsClassifier(n_neighbors=5)
            # Train the model on face data and labels
            knn.fit(faces, labels)
            # Save the trained model to a file
            joblib.dump(knn, 'static/face_recognition_model.pkl')

    # Function to extract attendance data from CSV
    def extract_attendance(self):
        try:
            # Read the attendance CSV file
            df = pd.read_csv(self.csv_path)
            print(f"Read CSV: \n{df}")  # Debug: Show CSV contents
            # Return lists of names, roll numbers, times, and total entries
            return df['Name'].tolist(), df['Roll'].tolist(), df['Time'].tolist(), len(df)
        # Handle errors reading CSV and return empty lists
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return [], [], [], 0

    # Function to add a new attendance entry
    def add_attendance(self, name):
        try:
            # Split name into username and user ID
            username = name.split('_')[0]
            userid = name.split('_')[1]
            # Get current time for attendance
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Backup the existing CSV file
            backup_path = f'Attendance/Attendance-{self.datetoday}_backup.csv'
            if os.path.exists(self.csv_path):
                shutil.copy(self.csv_path, backup_path)
                print(f"Backed up CSV to {backup_path}")
            
            # Read current CSV and append new entry if user ID is not already present
            df = pd.read_csv(self.csv_path)
            if int(userid) not in list(df['Roll']):
                # Create new entry as a DataFrame
                new_entry = pd.DataFrame([[username, int(userid), current_time]], columns=['Name', 'Roll', 'Time'])
                # Append new entry to existing DataFrame
                df = pd.concat([df, new_entry], ignore_index=True)
                # Save updated DataFrame to CSV
                df.to_csv(self.csv_path, index=False)
                print(f"Added attendance: {username}, {userid}, {current_time}")
            else:
                print(f"User ID {userid} already in attendance")
        # Handle errors during attendance addition
        except Exception as e:
            print(f"Error adding attendance: {e}")

    # Function to get list of all registered users
    def getallusers(self):
        # Get list of user folders
        userlist = os.listdir('static/faces')
        names, rolls = [], []
        # Split each user folder name into name and roll number
        for user in userlist:
            name, roll = user.split('_')
            names.append(name)
            rolls.append(roll)
        # Return user list, names, rolls, and total count
        return userlist, names, rolls, len(userlist)

# Create an instance of AttendanceSystem
attendance_system = AttendanceSystem()

# Route for the home page
@app.route('/')
def home():
    # Extract attendance data
    names, rolls, times, l = attendance_system.extract_attendance()
    print(f"Home route - Names: {names}, Rolls: {rolls}, Times: {times}, Length: {l}")  # Debug
    # Render home.html with attendance data and total registered users
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=attendance_system.totalreg(), datetoday2=attendance_system.datetoday2)

# Route to start face recognition and attendance
@app.route('/start', methods=['GET'])
def start():
    # Update the background image (mimicking original behavior)
    attendance_system.imgBackground = attendance_system.load_background()  # Reload background
    
    # Check if face recognition model exists; train if it doesn't and faces are available
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        if os.listdir('static/faces'):
            attendance_system.train_model()
        else:
            # If no faces are registered, return to home page with error message
            names, rolls, times, l = attendance_system.extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=attendance_system.totalreg(), datetoday2=attendance_system.datetoday2, mess='No faces registered. Please add a new face to continue.')

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    # Check if webcam opened successfully
    if not cap.isOpened():
        names, rolls, times, l = attendance_system.extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=attendance_system.totalreg(), datetoday2=attendance_system.datetoday2, mess='Could not access webcam.')
    
    # Set webcam resolution to 800x600
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    # Initialize variables for attendance timeout and tracking
    start_time = datetime.now()
    timeout = 10
    attendance_taken = False
    identified_person = None
    
    # Define offsets to center webcam feed in background (1280x720)
    x_offset = 240  # Horizontally centered
    y_offset = 60   # Adjusted vertically to fit 600px height
    
    # Run loop for up to 10 seconds to detect and recognize faces
    while (datetime.now() - start_time).seconds < timeout:
        ret, frame = cap.read()
        # Check if frame was captured successfully
        if not ret:
            print("Error: Failed to capture frame")
            break
        # Resize frame to 800x600 for consistency
        frame = cv2.resize(frame, (800, 600))
        print(f"Frame shape: {frame.shape}")  # Debug
        # Detect faces in the frame
        faces = attendance_system.extract_faces(frame)
        if len(faces) > 0:
            # Process the first detected face
            (x, y, w, h) = faces[0]
            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            # Draw rectangle for name label above face
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            # Resize detected face to 50x50 for recognition
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            # Identify the face using the trained model
            identified_person = attendance_system.identify_face(face.reshape(1, -1))[0]
            # Add attendance entry for identified person
            attendance_system.add_attendance(identified_person)
            attendance_taken = True
            # Display identified person's name on frame
            cv2.putText(frame, f'{identified_person}', (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            # Display attendance confirmation text
            cv2.putText(frame, 'Attendance Taken', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            try:
                # Overlay frame onto background image if dimensions match
                if attendance_system.imgBackground.shape == (720, 1280, 3) and frame.shape == (600, 800, 3):
                    imgBackground_copy = attendance_system.imgBackground.copy()
                    imgBackground_copy[y_offset:y_offset + 600, x_offset:x_offset + 800] = frame
                    cv2.imshow('Attendance', imgBackground_copy)
                else:
                    print(f"Invalid dimensions: Background {attendance_system.imgBackground.shape}, Frame {frame.shape}")
                    cv2.imshow('Attendance', frame)
            # Handle errors during frame display
            except Exception as e:
                print(f"Error displaying frame: {e}")
                cv2.imshow('Attendance', frame)
            # Display frame for 1 second before breaking
            cv2.waitKey(1000)
            break
        
        # Display remaining time on frame
        remaining = timeout - (datetime.now() - start_time).seconds
        cv2.putText(frame, f'Time Left: {remaining}s', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        try:
            # Overlay frame onto background image if dimensions match
            if attendance_system.imgBackground.shape == (720, 1280, 3) and frame.shape == (600, 800, 3):
                imgBackground_copy = attendance_system.imgBackground.copy()
                imgBackground_copy[y_offset:y_offset + 600, x_offset:x_offset + 800] = frame
                cv2.imshow('Attendance', imgBackground_copy)
            else:
                print(f"Invalid dimensions: Background {attendance_system.imgBackground.shape}, Frame {frame.shape}")
                cv2.imshow('Attendance', frame)
        # Handle errors during frame display
        except Exception as e:
            print(f"Error displaying frame: {e}")
            cv2.imshow('Attendance', frame)
        # Exit loop if Escape key is pressed
        if cv2.waitKey(1) == 27:
            break

    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    # Extract updated attendance data
    names, rolls, times, l = attendance_system.extract_attendance()
    print(f"Start route - Updated Names: {names}, Rolls: {rolls}, Times: {times}, Length: {l}")  # Debug
    # Set message based on whether attendance was taken
    mess = 'Attendance Taken Successfully' if attendance_taken else 'No Face Detected'
    # Render home page with updated attendance data and message
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=attendance_system.totalreg(), datetoday2=attendance_system.datetoday2, mess=mess)

# Route to add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    # Handle POST request for adding a new user
    if request.method == 'POST':
        # Get username and user ID from form
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        # Validate that username contains only letters and spaces
        if not newusername.replace(" ", "").isalpha():
            names, rolls, times, l = attendance_system.extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=attendance_system.totalreg(), datetoday2=attendance_system.datetoday2, mess='Username must contain only letters and spaces.')
        # Validate that user ID is a positive integer
        try:
            newuserid = int(newuserid)
            if newuserid <= 0:
                raise ValueError
        except ValueError:
            names, rolls, times, l = attendance_system.extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=attendance_system.totalreg(), datetoday2=attendance_system.datetoday2, mess='User ID must be a positive number.')
        
        # Check if user already exists
        userimagefolder = f'static/faces/{newusername}_{newuserid}'
        if os.path.isdir(userimagefolder):
            names, rolls, times, l = attendance_system.extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=attendance_system.totalreg(), datetoday2=attendance_system.datetoday2, mess='User already exists.')
        
        # Create folder for new user's face images
        os.makedirs(userimagefolder)
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        # Check if webcam opened successfully
        if not cap.isOpened():
            names, rolls, times, l = attendance_system.extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=attendance_system.totalreg(), datetoday2=attendance_system.datetoday2, mess='Could not access webcam.')
        
        # Set webcam resolution to 800x600
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        
        # Initialize counters for images captured
        i, j = 0, 0
        start_time = datetime.now()
        # Capture images for 30 seconds or until enough images are collected
        while (datetime.now() - start_time).seconds < 30:
            _, frame = cap.read()
            # Resize frame to 800x600 for consistency
            frame = cv2.resize(frame, (800, 600))
            # Detect faces in the frame
            faces = attendance_system.extract_faces(frame)
            for (x, y, w, h) in faces:
                # Draw rectangle around detected face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                # Display number of images captured
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
                # Save face image every 5 frames
                if j % 5 == 0:
                    name = f'{newusername}_{i}.jpg'
                    cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            # Break if enough images are captured
            if i >= nimgs:
                break
            # Display frame with annotations
            cv2.imshow('Adding new User', frame)
            # Exit loop if Escape key is pressed
            if cv2.waitKey(1) == 27:
                break
        # Release webcam and close windows
        cap.release()
        cv2.destroyAllWindows()
        
        # If not enough images were captured, remove user folder and show error
        if i < nimgs:
            shutil.rmtree(userimagefolder)
            names, rolls, times, l = attendance_system.extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=attendance_system.totalreg(), datetoday2=attendance_system.datetoday2, mess='Failed to capture enough images.')
        
        # Train model with new user data
        attendance_system.train_model()
        # Extract updated attendance data
        names, rolls, times, l = attendance_system.extract_attendance()
        # Render home page with success message
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=attendance_system.totalreg(), datetoday2=attendance_system.datetoday2, mess=f'User {newusername} added successfully.')

    # For GET request, render home page with current attendance data
    names, rolls, times, l = attendance_system.extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=attendance_system.totalreg(), datetoday2=attendance_system.datetoday2)

# Run the Flask app in debug mode if script is executed directly
if __name__ == '__main__':
    app.run(debug=True)