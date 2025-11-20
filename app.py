import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
from PIL import Image
import pandas as pd
import os

st.set_page_config(page_title="AI Attendance System", layout="wide")
st.title("AI Attendance System")

# Create folders if they don't exist
if not os.path.exists("students"):
    os.makedirs("students")

if "attendance" not in st.session_state:
    st.session_state["attendance"] = {}

# -----------------------
# Register Single Student
# -----------------------
st.header("Register Student")
name = st.text_input("Student Name")
roll = st.text_input("Roll Number")
img_file = st.file_uploader("Upload Student Photo (jpg/png)", type=["jpg", "png"])

if st.button("Register Student"):
    if name and roll and img_file:
        img = Image.open(img_file)
        img_path = f"students/{roll}_{name}.jpg"
        img.save(img_path)
        st.success(f"Student {name} registered successfully!")
    else:
        st.error("Please fill all fields and upload a photo.")

# -----------------------
# Take Attendance
# -----------------------
st.header("Take Attendance")

cam_option = st.selectbox("Use Camera?", ["Yes", "No"])

if cam_option == "Yes":
    run = st.button("Start Camera")
    if run:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture from camera.")
                break

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(rgb_frame, channels="RGB")

            # Press 'q' to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

else:
    uploaded_img = st.file_uploader("Upload Image for Attendance", type=["jpg", "png"])
    if uploaded_img:
        uploaded_img_pil = Image.open(uploaded_img)
        uploaded_img_cv = cv2.cvtColor(np.array(uploaded_img_pil), cv2.COLOR_RGB2BGR)

        # Loop through registered students
        for student_file in os.listdir("students"):
            student_img_path = f"students/{student_file}"
            result = DeepFace.verify(uploaded_img_cv, student_img_path, enforce_detection=False)
            if result["verified"]:
                st.success(f"Attendance marked for {student_file}")
                st.session_state["attendance"][student_file] = "Present"
            else:
                st.info(f"{student_file} not recognized")

# -----------------------
# Show Attendance
# -----------------------
st.header("Attendance List")
if st.session_state["attendance"]:
    df = pd.DataFrame.from_dict(st.session_state["attendance"], orient="index", columns=["Status"])
    st.table(df)
else:
    st.info("No attendance recorded yet.")
