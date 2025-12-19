import cv2
import os
import json
import numpy as np
from deepface import DeepFace
from datetime import datetime

RTSP_URL = "rtsp://admin:Nodo2023@192.168.1.213:554/dev.hik-connect.com/channels/101/"
MODEL_NAME = "Facenet"
DISTANCE_THRESHOLD = 0.65
OUTPUT_DIR = "faces"
JSON_FILE = "people.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

people = []
person_id = 1


def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_person(embedding):
    for p in people:
        dist = cosine_distance(embedding, p["embedding"])
        if dist < DISTANCE_THRESHOLD:
            return p
    return None


print("🎥 Conectando al RTSP...")
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("❌ No se pudo abrir el stream RTSP")
    exit()

print("✅ Stream conectado")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 5 != 0:
        continue  # baja carga CPU

    try:
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="retinaface",
            enforce_detection=False
        )
    except:
        continue

    for face in faces:
        face_img = face["face"]
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
        person = find_person(embedding)
        now = datetime.now().isoformat()

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

    print(f"👥 Personas detectadas: {len(people)}", end="\r")

# Guardar JSON
for p in people:
    del p["embedding"]

with open(JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(people, f, indent=2, ensure_ascii=False)

cap.release()
print("\n✅ Proceso finalizado")
print(f"📄 Archivo generado: {JSON_FILE}")
