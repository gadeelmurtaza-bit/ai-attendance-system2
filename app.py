import streamlit as st
import cv2
import os
import numpy as np
from deepface import DeepFace
from PIL import Image
import pandas as pd
import time

STUDENT_DIR = "students"
if not os.path.exists(STUDENT_DIR):
    os.makedirs(STUDENT_DIR)

ATTENDANCE_FILE = "attendance.csv"

def save_student(name, roll, image):
    filename = f"{STUDENT_DIR}/{roll}_{name}.jpg"
    image.save(filename)
    return filename

def get_all_students():
    files = os.listdir(STUDENT_DIR)
    students = []
    for f in files:
        if f.endswith(".jpg"):
            roll, name = f.replace(".jpg","").split("_", 1)
            students.append({"roll": roll, "name": name, "file": f})
    return students

def mark_attendance(roll, name):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[roll, name, ts]], columns=["Roll", "Name", "Time"])
    
    if os.path.exists(ATTENDANCE_FILE):
        df.to_csv(ATTENDANCE_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(ATTENDANCE_FILE, index=False)

def detect_face_match(captured_img):
    students = get_all_students()

    for student in students:
        db_img_path = f"{STUDENT_DIR}/{student['file']}"

        try:
            result = DeepFace.verify(img1_path=captured_img, img2_path=db_img_path, enforce_detection=False)
            if result["verified"]:
                return student["roll"], student["name"]
        except:
            pass

    return None, None

st.title("AI Attendance System (DeepFace Version)")

menu = st.sidebar.selectbox("Menu", ["Register Student", "Take Attendance", "View Attendance"])

if menu == "Register Student":
    st.header("Register New Student")
    name = st.text_input("Enter Name")
    roll = st.text_input("Enter Roll Number")
    uploaded = st.file_uploader("Upload Student Photo", type=["jpg", "jpeg", "png"])

    if uploaded and name and roll:
        img = Image.open(uploaded)
        save_student(name, roll, img)
        st.success("Student Registered Successfully!")

elif menu == "Take Attendance":
    st.header("Take Attendance with Webcam")

    camera = st.camera_input("Capture Image")

    if camera:
        captured_img = Image.open(camera)
        captured_img.save("temp.jpg")

        roll, name = detect_face_match("temp.jpg")

        if roll:
            mark_attendance(roll, name)
            st.success(f"Attendance Marked: {name} ({roll})")
        else:
            st.error("Face not matched with any student.")

elif menu == "View Attendance":
    st.header("Attendance Records")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)
    else:
        st.info("No attendance taken yet.")
