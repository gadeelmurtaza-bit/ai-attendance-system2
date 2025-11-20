import face_recognition
import numpy as np
import cv2

def extract_encoding(image_path):
    img = face_recognition.load_image_file(image_path)
    enc = face_recognition.face_encodings(img)
    return enc[0] if len(enc) > 0 else None

def compare_faces(known_encoding, unknown_encoding, tolerance=0.45):
    return face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=tolerance)[0]
