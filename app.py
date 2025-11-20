import streamlit as st
from deepface import DeepFace
import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd

st.title("AI Attendance System")

# Create folder to store student images
if not os.path.exists("students"):
    os.makedirs("students")

# Sidebar
menu = ["Home", "Add Single Student", "Bulk Add Students", "Take Attendance"]
choice = st.sidebar.selectbox("Menu", menu)

# --- Home ---
if choice == "Home":
    st.subheader("Welcome to AI Attendance System")
    st.text("Use the sidebar to add students or take attendance.")

# --- Add Single Student ---
elif choice == "Add Single Student":
    st.subheader("Add Single Student")
    name = st.text_input("Student Name")
    roll_no = st.text_input("Roll Number")
    uploaded_file = st.file_uploader("Upload Student Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if st.button("Save Student"):
        if name and roll_no and uploaded_file:
            file_path = f"students/{roll_no}_{name}.jpg"
            image = Image.open(uploaded_file)
            image.save(file_path)
            st.success(f"Student {name} saved successfully!")
        else:
            st.error("Please provide name, roll number, and photo.")

# --- Bulk Add Students ---
elif choice == "Bulk Add Students":
    st.subheader("Bulk Add Students")
    st.text("Upload multiple student photos in a folder and name them as RollName.jpg")
    uploaded_files = st.file_uploader("Upload Multiple Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if st.button("Save Students"):
        for file in uploaded_files:
            file_path = f"students/{file.name}"
            image = Image.open(file)
            image.save(file_path)
        st.success(f"{len(uploaded_files)} students saved successfully!")

# --- Take Attendance ---
elif choice == "Take Attendance":
    st.subheader("Take Attendance")

    run = st.button("Start Camera")
    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        attendance = pd.DataFrame(columns=["Roll No", "Name"])

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not detected.")
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Compare with all student images
            for student_img in os.listdir("students"):
                student_path = os.path.join("students", student_img)
                try:
                    result = DeepFace.verify(rgb_frame, student_path, enforce_detection=False)
                    if result["verified"]:
                        roll_no, name_with_ext = student_img.split("_")
                        name = os.path.splitext(name_with_ext)[0]
                        if not ((attendance['Roll No'] == roll_no) & (attendance['Name'] == name)).any():
                            attendance = pd.concat([attendance, pd.DataFrame([[roll_no, name]], columns=["Roll No", "Name"])], ignore_index=True)
                        cv2.putText(frame, f"{name} Present", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                except:
                    continue

            stframe.image(frame, channels="BGR")

        cap.release()
        st.dataframe(attendance)
