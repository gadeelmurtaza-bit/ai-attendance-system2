import streamlit as st
import numpy as np
import cv2
import os
from database import *
from utils.face_utils import extract_encoding, compare_faces
from utils.camera import capture_frame
import face_recognition

os.makedirs("data/images", exist_ok=True)
init_db()

st.set_page_config(page_title="AI Attendance System", layout="wide")

st.title("📚 AI Attendance System (SQLite Version)")

menu = ["Register Single Student", "Bulk Registration", "Take Attendance", "View Attendance"]
choice = st.sidebar.selectbox("Menu", menu)

# ------------------------------------------------------------
# 1️⃣ SINGLE STUDENT REGISTRATION
# ------------------------------------------------------------
if choice == "Register Single Student":
    st.header("Register a Student")

    name = st.text_input("Name")
    roll = st.text_input("Roll No")
    file = st.file_uploader("Upload Student Image (JPG)", type=["jpg", "jpeg"])

    if st.button("Register"):
        if name == "" or roll == "" or file is None:
            st.error("Please fill all fields!")
        else:
            img_path = f"data/images/{roll}.jpg"
            with open(img_path, "wb") as f:
                f.write(file.getbuffer())

            encoding = extract_encoding(img_path)

            if encoding is None:
                st.error("Face not detected. Upload another image.")
            else:
                add_student(name, roll, img_path, encoding.tobytes())
                st.success("Student Registered Successfully!")

# ------------------------------------------------------------
# 2️⃣ BULK REGISTRATION
# ------------------------------------------------------------
elif choice == "Bulk Registration":
    st.header("Bulk Registration")

    files = st.file_uploader("Upload multiple JPG images", type=["jpg"], accept_multiple_files=True)

    if st.button("Upload All"):
        for file in files:
            roll = file.name.split(".")[0]
            name = roll  # You can change this logic

            img_path = f"data/images/{file.name}"
            with open(img_path, "wb") as f:
                f.write(file.getbuffer())

            encoding = extract_encoding(img_path)
            if encoding is not None:
                add_student(name, roll, img_path, encoding.tobytes())

        st.success("Bulk Registration Completed!")

# ------------------------------------------------------------
# 3️⃣ ATTENDANCE SYSTEM (WEBCAM)
# ------------------------------------------------------------
elif choice == "Take Attendance":
    st.header("Take Attendance")

    if st.button("Start Attendance"):

        frame = capture_frame()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        unknown_locations = face_recognition.face_locations(rgb)
        unknown_encodings = face_recognition.face_encodings(rgb, unknown_locations)

        students = get_all_students()

        marked = []

        for unk_enc in unknown_encodings:
            for name, roll, path, enc_blob in students:
                known_enc = np.frombuffer(enc_blob, dtype=np.float64)

                if compare_faces(known_enc, unk_enc):
                    mark_attendance(roll, name, "Present")
                    marked.append((roll, name))

        if len(marked) == 0:
            st.warning("No known student detected!")

        st.success("Attendance Marked!")
        st.write(marked)

# ------------------------------------------------------------
# 4️⃣ VIEW ATTENDANCE LOG
# ------------------------------------------------------------
elif choice == "View Attendance":
    st.header("Attendance Logs")

    data = get_attendance()

    if len(data) == 0:
        st.info("No attendance records yet.")
    else:
        st.write(data)
