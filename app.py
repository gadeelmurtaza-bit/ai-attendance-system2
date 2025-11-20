import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from datetime import datetime
from PIL import Image

# --------- CREATE FOLDERS -----------
if not os.path.exists("students"):
    os.makedirs("students")

if not os.path.exists("database.csv"):
    df = pd.DataFrame(columns=["name", "roll", "image_path"])
    df.to_csv("database.csv", index=False)


# --------- LOAD DATABASE -----------
def load_database():
    return pd.read_csv("database.csv")


# --------- SAVE NEW STUDENT ----------
def save_student(name, roll, image):
    filename = f"students/{roll}.jpg"
    image.save(filename)

    df = load_database()
    df.loc[len(df)] = [name, roll, filename]
    df.to_csv("database.csv", index=False)


# --------- FACE RECOGNITION ----------
def match_face(img):
    df = load_database()
    for index, row in df.iterrows():
        db_image = row["image_path"]

        try:
            result = DeepFace.verify(img_path=img, db_path=db_image, detector_backend='opencv')
            if result["verified"]:
                return row["name"], row["roll"]
        except:
            pass

    return None, None


# --------- ATTENDANCE SAVE ----------
def save_attendance(name, roll):
    filename = "attendance.csv"

    if not os.path.exists(filename):
        pd.DataFrame(columns=["name", "roll", "time"]).to_csv(filename, index=False)

    df = pd.read_csv(filename)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df.loc[len(df)] = [name, roll, now]
    df.to_csv(filename, index=False)


# ---------- STREAMLIT UI -----------
st.title("AI Attendance System (DeepFace)")

menu = st.sidebar.selectbox(
    "Menu",
    ["Single Student Register", "Bulk Register", "Take Attendance", "View Database", "View Attendance"]
)

# ---------- Single Registration ----------
if menu == "Single Student Register":
    st.header("Register a Student")

    name = st.text_input("Student Name")
    roll = st.text_input("Roll Number")
    file = st.file_uploader("Upload Student Photo (JPG)", type=["jpg", "jpeg"])

    if st.button("Register"):
        if name and roll and file:
            img = Image.open(file)
            save_student(name, roll, img)
            st.success("Student Registered Successfully!")
        else:
            st.error("Please provide all details.")

# ---------- Bulk Registration ----------
elif menu == "Bulk Register":
    st.header("Bulk Register Students")
    st.markdown("Upload multiple student images named as: **rollnumber_name.jpg**")

    files = st.file_uploader("Upload Image Files", type=["jpg"], accept_multiple_files=True)

    if st.button("Upload All"):
        count = 0
        for f in files:
            filename = f.name.split(".")[0]
            try:
                roll, name = filename.split("_")
                img = Image.open(f)
                save_student(name, roll, img)
                count += 1
            except:
                continue
        st.success(f"{count} Students Registered Successfully!")

# ---------- Take Attendance ----------
elif menu == "Take Attendance":
    st.header("Take Attendance via Live Camera")

    start = st.button("Start Camera")

    if start:
        cap = cv2.VideoCapture(0)

        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not found.")
                break

            # Display
            stframe.image(frame, channels="BGR")

            # Convert
            img_path = "live_capture.jpg"
            cv2.imwrite(img_path, frame)

            # Match
            name, roll = match_face(img_path)

            if name is not None:
                save_attendance(name, roll)
                st.success(f"Attendance Marked: {name} ({roll})")
                break

        cap.release()

# ---------- View Database ----------
elif menu == "View Database":
    st.header("Registered Students")
    df = load_database()
    st.dataframe(df)

# ---------- View Attendance ----------
elif menu == "View Attendance":
    st.header("Attendance Log")

    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        st.dataframe(df)
    else:
        st.info("No attendance recorded yet.")
