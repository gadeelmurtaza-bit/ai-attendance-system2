import streamlit as st
import os
import pandas as pd
from deepface import DeepFace
from PIL import Image
from datetime import datetime

# ------------ SETUP -------------
if not os.path.exists("students"):
    os.makedirs("students")

if not os.path.exists("database.csv"):
    pd.DataFrame(columns=["name", "roll", "image_path"]).to_csv("database.csv", index=False)


def load_database():
    return pd.read_csv("database.csv")


def save_student(name, roll, image):
    path = f"students/{roll}.jpg"
    image.save(path)
    df = load_database()
    df.loc[len(df)] = [name, roll, path]
    df.to_csv("database.csv", index=False)


def match_face(img):
    df = load_database()
    for _, row in df.iterrows():
        try:
            result = DeepFace.verify(
                img1_path=img,
                img2_path=row["image_path"],
                detector_backend='retinaface'
            )
            if result["verified"]:
                return row["name"], row["roll"]
        except:
            pass
    return None, None


def save_attendance(name, roll):
    if not os.path.exists("attendance.csv"):
        pd.DataFrame(columns=["name", "roll", "time"]).to_csv("attendance.csv", index=False)

    df = pd.read_csv("attendance.csv")
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[len(df)] = [name, roll, time_now]
    df.to_csv("attendance.csv", index=False)


# ------------ UI -------------
st.title("AI Attendance System (DeepFace — No OpenCV)")

menu = st.sidebar.selectbox(
    "Menu",
    ["Register Single Student", "Bulk Register Students", "Take Attendance", "View Database", "View Attendance"]
)

# ---------- Register Single ----------
if menu == "Register Single Student":
    st.header("Register Student")

    name = st.text_input("Name")
    roll = st.text_input("Roll Number")
    uploaded = st.file_uploader("Upload Photo (JPG)", type=["jpg", "jpeg"])

    if st.button("Register"):
        if name and roll and uploaded:
            img = Image.open(uploaded)
            save_student(name, roll, img)
            st.success("Student registered successfully!")
        else:
            st.error("Fill all fields.")

# ---------- Bulk Register ----------
elif menu == "Bulk Register Students":
    st.header("Bulk Register (Format: roll_name.jpg)")
    files = st.file_uploader("Upload multiple images", accept_multiple_files=True, type=["jpg"])

    if st.button("Upload All"):
        count = 0
        for f in files:
            try:
                base = f.name.split(".")[0]
                roll, name = base.split("_")
                img = Image.open(f)
                save_student(name, roll, img)
                count += 1
            except:
                pass
        st.success(f"{count} students registered successfully!")

# ---------- Take Attendance ----------
elif menu == "Take Attendance":
    st.header("Take Attendance via Camera")

    img_file = st.camera_input("Capture Image")

    if img_file:
        captured = Image.open(img_file)
        temp_path = "temp_capture.jpg"
        captured.save(temp_path)

        name, roll = match_face(temp_path)

        if name:
            save_attendance(name, roll)
            st.success(f"Attendance Marked: {name} ({roll})")
        else:
            st.error("Face not recognized!")

# ---------- View DB ----------
elif menu == "View Database":
    st.header("Student Database")
    st.dataframe(load_database())

# ---------- View Attendance ----------
elif menu == "View Attendance":
    st.header("Attendance Log")
    if os.path.exists("attendance.csv"):
        st.dataframe(pd.read_csv("attendance.csv"))
    else:
        st.info("No attendance recorded yet.")
