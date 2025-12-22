import cv2
import os
import json
import numpy as np
from deepface import DeepFace
from datetime import datetime

# ================= CONFIG =================
RTSP_URL = "rtsp://admin:Nodo2023@192.168.1.213:554/dev.hik-connect.com/channels/101/"
OUTPUT_DIR = "faces"
JSON_FILE = "people.json"

MODEL_NAME = "Facenet"
DIST_THRESHOLD = 0.6
FRAME_SKIP = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================

people = []
person_id = 1
frame_count = 0

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------- utils ----------
def cosine(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def match_person(embedding):
    for p in people:
        if cosine(embedding, p["embedding"]) < DIST_THRESHOLD:
            return p
    return None

# ---------- RTSP ----------
print("🎥 Conectando al RTSP...")
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("❌ No se pudo abrir el stream")
    exit()

print("✅ Stream conectado")

# ---------- LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            continue

        try:
            rep = DeepFace.represent(
                img_path=face_img,
                model_name=MODEL_NAME,
                enforce_detection=False
            )
        except:
            continue

        embedding = np.array(rep[0]["embedding"])
        now = datetime.now().isoformat()

        person = match_person(embedding)

        if person:
            person["last_seen"] = now
            person["appearances"] += 1
        else:
            pid = person_id
            person_id += 1

            person_dir = os.path.join(OUTPUT_DIR, f"person_{pid}")
            os.makedirs(person_dir, exist_ok=True)

            img_path = os.path.join(person_dir, "face.jpg")
            cv2.imwrite(img_path, face_img)

            people.append({
                "id": pid,
                "first_seen": now,
                "last_seen": now,
                "appearances": 1,
                "embedding": embedding,
                "image": img_path
            })

    print(f"👥 Personas únicas: {len(people)}", end="\r")

# ---------- SAVE ----------
for p in people:
    del p["embedding"]

with open(JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(people, f, indent=2, ensure_ascii=False)

cap.release()
print("\n✅ Finalizado")
print(f"📄 {JSON_FILE} generado")
