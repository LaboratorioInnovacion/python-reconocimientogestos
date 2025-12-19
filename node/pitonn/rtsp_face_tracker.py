import cv2
import numpy as np
import json
import time
from deepface import DeepFace
from datetime import datetime

# =============================
# CONFIG
# =============================
RTSP_URL = "rtsp://admin:Nodo2023@192.168.1.213:554/dev.hik-connect.com/channels/101/"
MODEL_NAME = "Facenet"
DISTANCE_THRESHOLD = 0.55
JSON_OUTPUT = "people_rtsp.json"
PROCESS_EVERY_N_FRAMES = 10

# =============================
# FACE DETECTOR
# =============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =============================
# STATE
# =============================
people = []
person_id = 1
frame_count = 0


def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embedding(face_img):
    try:
        rep = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            enforce_detection=False
        )
        return np.array(rep[0]["embedding"])
    except:
        return None


def find_person(embedding):
    for p in people:
        dist = cosine_distance(embedding, p["embedding"])
        if dist < DISTANCE_THRESHOLD:
            return p
    return None


def save_json():
    data = []
    for p in people:
        data.append({
            "id": p["id"],
            "first_seen": p["first_seen"],
            "last_seen": p["last_seen"],
            "appearances": p["count"]
        })

    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =============================
# RTSP STREAM
# =============================
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("❌ No se pudo abrir RTSP")
    exit()

print("📡 RTSP conectado — tracking iniciado")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame perdido")
        break

    frame_count += 1

    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        embedding = get_embedding(face_img)

        if embedding is None:
            continue

        person = find_person(embedding)

        if person:
            person["last_seen"] = datetime.now().isoformat()
            person["count"] += 1
            pid = person["id"]
        else:
            people.append({
                "id": person_id,
                "embedding": embedding,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "count": 1
            })
            pid = person_id
            person_id += 1

        # Dibujar
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {pid}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("RTSP Face Tracker", frame)

    save_json()

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Finalizado")
