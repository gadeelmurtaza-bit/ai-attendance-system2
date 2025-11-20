# app.py
import streamlit as st
import face_recognition
from PIL import Image
import numpy as np
import os
import pandas as pd
import cv2
from datetime import datetime

# ----------------- Setup -----------------
st.set_page_config(page_title="AI Attendance System", layout="wide")
st.title("AI Attendance System")

# Create folders if not exist
if not os.path.exists("students"):
    os.makedirs("students")
if not os.path.exists("attendance"):
    os.makedirs("attendance")

# ----------------- Helper Functions -----------------
def save_student(name, roll, img):
    path = f"students/{roll}_{name}.jpg"
    img.save(path)
    st.success(f"Saved student {name} with roll {roll}")

def load_known_faces():
    known_encodings = []
    known_names = []
    for file in os.listdir("students"):
        img_path = os.path.join("students", file)
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            name = file.split("_",1)[1].replace(".jpg","")
            known_names.append(name)
    return known_encodings, known_names

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"attendance/{today}.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=["Name", "Time"])
    
    if name not in df["Name"].values:
        df = df.append({"Name": name, "Time": datetime.now().strftime("%H:%M:%S")}, ignore_index=True)
        df.to_csv(filename, index=False)

# ----------------- Sidebar -----------------
mode = st.sidebar.selectbox("Mode", ["Single Registration", "Bulk Registration", "Take Attendance"])

# ----------------- Single Registration -----------------
if mode == "Single Registration":
    st.subheader("Single Student Registration")
    name = st.text_input("Name")
    roll = st.text_input("Roll Number")
    img_file = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])
    
    if st.button("Register Student"):
        if name and roll and img_file:
            img = Image.open(img_file)
            save_student(name, roll, img)
        else:
            st.error("Please fill all fields!")

# ----------------- Bulk Registration -----------------
elif mode == "Bulk Registration":
    st.subheader("Bulk Student Registration")
    csv_file = st.file_uploader("Upload CSV (name,roll)", type=["csv"])
    img_zip = st.file_uploader("Upload ZIP of images", type=["zip"])
    
    st.info("Bulk registration is possible but must manually extract images in 'students/' folder matching roll_number_name.jpg")

# ----------------- Take Attendance -----------------
elif mode == "Take Attendance":
    st.subheader("Take Attendance")
    st.info("Webcam will open, look at the camera to mark attendance.")

    known_encodings, known_names = load_known_faces()
    if len(known_encodings) == 0:
        st.warning("No students registered yet!")
    else:
        run = st.button("Start Camera")
        FRAME_WINDOW = st.image([])

        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Could not access webcam")
                break

            small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    mark_attendance(name)
                    top, right, bottom, left = face_location
                    top *= 4; right *= 4; bottom *= 4; left *= 4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                    cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        cv2.destroyAllWindows()
