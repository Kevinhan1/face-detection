import streamlit as st
import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model dan detektor wajah
model = load_model("model_beautiful_ugly2.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Title
st.title("💁‍♂️ Face Beauty Detection (Handsome or Ugly)")

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar wajah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Buka gambar
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        st.warning("❗Tidak ditemukan wajah.")
    else:
        for (x, y, w, h) in faces:
            # Landmark
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), rect)
            landmarks = [(pt.x, pt.y) for pt in shape.parts()]
            landmarks_flat = np.array(landmarks).flatten().astype(np.float32)
            landmark_input = np.expand_dims(landmarks_flat, axis=0)

            # Crop wajah dan resize
            face_crop = img[y:y + h, x:x + w]
            try:
                resized_face = cv2.resize(face_crop, (128, 128)) / 255.0
            except:
                st.warning("❗Gagal resize wajah.")
                continue

            img_input = np.expand_dims(resized_face.astype(np.float32), axis=0)

            # Prediksi
            pred = model.predict([img_input, landmark_input])[0][0]
            label = "Handsome" if pred >= 0.7 else "Ugly"
            color = (0, 255, 0) if pred >= 0.5 else (0, 0, 255)

            # Gambar kotak & teks
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} ({pred:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Gambar landmark
            for (lx, ly) in landmarks:
                cv2.circle(img, (lx, ly), 1, (255, 255, 0), -1)

        # Tampilkan hasil
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi", use_column_width=True)
