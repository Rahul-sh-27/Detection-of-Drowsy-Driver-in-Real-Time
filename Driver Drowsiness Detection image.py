import tkinter as tk
from tkinter import messagebox, PhotoImage
import os
import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords

# Path to the file where user credentials will be stored
CREDENTIALS_FILE = 'credentials.txt'

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/basha/Downloads/Driver-Drowsiness-Detection-master/dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# Function to run the drowsy driver detection system
def run_detection_system():
    input_image_path = 'C:/Users/basha/Desktop/image4.jpg'
    input_img = cv2.imread(input_image_path)

    if input_img is None:
        print("[ERROR] Could not read input image.")
        exit()

    output_width = 1024
    input_img = imutils.resize(input_img, width=output_width)
    frame = input_img.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    image_points = np.array([
        (359, 391), (399, 561), (337, 297), (513, 301),
        (345, 465), (453, 469)
    ], dtype="double")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    EYE_AR_THRESH = 0.25
    MOUTH_AR_THRESH = 0.79
    EYE_AR_CONSEC_FRAMES = 3
    COUNTER = 0

    (mStart, mEnd) = (49, 68)
    rects = detector(gray, 0)

    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for rect in rects:
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Alert!!! Eyes Closed! Driver is Drowsy", (500, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0

        mouth = shape[mStart:mEnd]
        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (520, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Alert!!! Yawning! Driver is Drowsy", (660, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for (i, (x, y)) in enumerate(shape):
            if i == 33:
                image_points[0] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 8:
                image_points[1] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 36:
                image_points[2] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 45:
                image_points[3] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 48:
                image_points[4] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 54:
                image_points[5] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        (head_tilt_degree, start_point, end_point, end_point_alt) = getHeadTiltAndCoords(size, image_points, frame.shape[0])
        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

        if head_tilt_degree:
            cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Original Image", input_img)
    cv2.imshow("Detected Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to handle sign-up
def signup():
    def save_user():
        username = new_username_entry.get()
        password = new_password_entry.get()
        
        if username and password:
            # Append new user credentials to the file
            with open(CREDENTIALS_FILE, 'a') as file:
                file.write(f"{username}:{password}\n")
            
            messagebox.showinfo("Sign Up Successful", "Account created successfully!")
            signup_window.destroy()
        else:
            messagebox.showerror("Sign Up Failed", "Please fill out all fields")

    # Create the sign-up window
    signup_window = tk.Toplevel(root)
    signup_window.title("Sign Up")
    signup_window.geometry("400x300")
    signup_window.configure(bg="#f0f0f0")

    # Keep a reference to the logo image
    logo = PhotoImage(file='C:/Users/basha/Desktop/1.png')  # Replace with your logo path
    tk.Label(signup_window, image=logo, bg="#f0f0f0").pack(pady=10)
    signup_window.logo = logo  # Keep a reference to the logo

    tk.Label(signup_window, text="Create a New Account", font=("Arial", 16), bg="#f0f0f0").pack(pady=10)
    tk.Label(signup_window, text="Username:", bg="#f0f0f0").pack(pady=5)
    new_username_entry = tk.Entry(signup_window)
    new_username_entry.pack(pady=5)
    tk.Label(signup_window, text="Password:", bg="#f0f0f0").pack(pady=5)
    new_password_entry = tk.Entry(signup_window, show="*")
    new_password_entry.pack(pady=5)
    tk.Button(signup_window, text="Sign Up", command=save_user).pack(pady=10)

# Function to validate login
def validate_login():
    username = username_entry.get()
    password = password_entry.get()

    if username and password:
        # Read credentials from the file
        if os.path.exists(CREDENTIALS_FILE):
            with open(CREDENTIALS_FILE, 'r') as file:
                credentials = file.readlines()
                credentials = [line.strip().split(':') for line in credentials]
                credentials_dict = {cred[0]: cred[1] for cred in credentials}
                
                # Check if the username and password match
                if username in credentials_dict and credentials_dict[username] == password:
                    messagebox.showinfo("Login Successful", "Welcome, " + username + "!")
                    root.destroy()  # Close the login window
                    run_detection_system()  # Run the detection system
                else:
                    messagebox.showerror("Login Failed", "Invalid username or password")
        else:
            messagebox.showerror("Login Failed", "No users found. Please sign up first.")
    else:
        messagebox.showerror("Login Failed", "Please enter both username and password")

root = tk.Tk()
root.title("Login Page")
root.geometry("800x600")
root.configure(bg='lightblue')

# Load logo image
logo = PhotoImage(file="C:/Users/basha/Desktop/1.png")  # Update the path to your logo image
logo = logo.subsample(3, 3)  # Resize the logo to be smaller

# Load additional images for the corners
left_photo = PhotoImage(file="C:/Users/basha/Desktop/bs.png")  # Update with your left corner image path
right_photo = PhotoImage(file="C:/Users/basha/Desktop/nns.png")  # Update with your right corner image path

# Resize the corner images to passport size
left_photo = left_photo.subsample(4, 4)
right_photo = right_photo.subsample(4, 4)

# Add college name and department name above the logo
tk.Label(root, text="SJC Institute of Technology", bg='orange', font=('Helvetica', 20, 'bold')).pack(pady=5)
tk.Label(root, text="Department of Computer Science and Engineering", bg='orange', font=('Helvetica', 18)).pack(pady=5)

# Add logo to login window
logo_label = tk.Label(root, image=logo, bg='lightblue')
logo_label.pack(pady=10)

# Add photos at the top corners
left_photo_label = tk.Label(root, image=left_photo, bg='lightblue')
left_photo_label.place(x=10, y=10)
right_photo_label = tk.Label(root, image=right_photo, bg='lightblue')
right_photo_label.place(x=1100, y=10)  # Adjusted x-coordinate to fit within 800x600 window

tk.Label(root, text="Username", bg='lightblue', font=('Helvetica', 16)).pack(pady=5)
username_entry = tk.Entry(root, font=('Helvetica', 14))
username_entry.pack(pady=5)

tk.Label(root, text="Password", bg='lightblue', font=('Helvetica', 16)).pack(pady=5)
password_entry = tk.Entry(root, show="*", font=('Helvetica', 14))
password_entry.pack(pady=5)

login_button = tk.Button(root, text="Login", command=validate_login, font=('Helvetica', 16))
login_button.pack(pady=10)

signup_button = tk.Button(root, text="SigSn Up", command=signup, font=('Helvetica', 16))
signup_button.pack(pady=10)

# Add your name and guide's name below the login text area
tk.Label(root, text="By: Shaik Al Kashifah", bg='lightblue', font=('Helvetica', 14)).pack(pady=5)
tk.Label(root, text="Under the guidance of: Dr. Seshaiah Merikapudi", bg='lightblue', font=('Helvetica', 14)).pack(pady=5)

root.mainloop()
