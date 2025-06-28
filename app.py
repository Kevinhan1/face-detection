import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model

# Load model CNN
model = load_model("model_beautiful_ugly2.h5")

# Load Haar Cascade dan Dlib
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Buka kamera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Gagal membuka kamera.")
    exit()

print("✅ Kamera aktif. Tekan 'q' untuk keluar.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("❌ Gagal membaca frame dari kamera.")
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(img, 1.1, 4)

    for (x, y, w, h) in faces:
        rect = dlib.rectangle(x, y, x + w, y + h)
        shape = predictor(img, rect)
        landmarks = [(pt.x, pt.y) for pt in shape.parts()]
        landmarks_flat = np.array(landmarks).flatten().astype(np.float32)
        landmark_input = np.expand_dims(landmarks_flat, axis=0)

        face_crop = img[y:y + h, x:x + w]
        try:
            resized_face = cv2.resize(face_crop, (128, 128)) / 255.0
        except:
            continue
        img_input = np.expand_dims(resized_face.astype(np.float32), axis=0)

        pred = model.predict([img_input, landmark_input])[0][0]
        label = "Handsome" if pred >= 0.7 else "Ugly"
        color = (0, 255, 0) if pred >= 0.5 else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} ({pred:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Handsome and Ugly Detection", frame)

    # Tekan tombol 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
