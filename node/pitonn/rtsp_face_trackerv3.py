import cv2
import os
import json
import time
import numpy as np
import signal
import sys
from deepface import DeepFace
from datetime import datetime

RTSP_URL = "rtsp://admin:Nodo2023@192.168.1.213:554/dev.hik-connect.com/channels/101/"
OUTPUT_DIR = "faces"
JSON_FILE = "people.json"

MODEL_NAME = "Facenet"
DISTANCE_THRESHOLD = 0.55
FRAME_SKIP = 8

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
    print("❌ No se pudo abrir el stream")
    # Intentar reconectar unas veces antes de salir
    retries = 0
    while retries < 5 and not cap.isOpened():
        time.sleep(2)
        cap = cv2.VideoCapture(RTSP_URL)
        retries += 1
    if not cap.isOpened():
        print("❌ No se pudo abrir el stream tras varios reintentos. Abortando.")
        sys.exit(1)

print("✅ Stream conectado")

frame_count = 0

def save_people():
    # Guardar JSON sin embeddings (no serializables)
    try:
        dump_list = []
        for p in people:
            copy_p = p.copy()
            if "embedding" in copy_p:
                del copy_p["embedding"]
            dump_list.append(copy_p)
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(dump_list, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Archivo actualizado: {JSON_FILE}")
    except Exception as e:
        print(f"Error guardando JSON: {e}")


def try_reconnect(max_attempts=10, delay=2):
    print("⚠️  Conexión perdida. Intentando reconectar...")
    cap.release()
    attempts = 0
    while attempts < max_attempts:
        time.sleep(delay)
        new_cap = cv2.VideoCapture(RTSP_URL)
        if new_cap.isOpened():
            print("🔁 Reconectado al stream")
            return new_cap
        attempts += 1
        print(f"  intento {attempts}/{max_attempts}...")
    print("❌ No se pudo reconectar después de varios intentos.")
    return None


def handle_exit(signum, frame):
    print("\n🏁 Señal de salida recibida, guardando y cerrando...")
    save_people()
    try:
        cap.release()
    except:
        pass
    sys.exit(0)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

while True:
    ret, frame = cap.read()
    if not ret:
        # intentar reconectar
        new_cap = try_reconnect()
        if new_cap is None:
            break
        cap = new_cap
        ret, frame = cap.read()
        if not ret:
            # si sigue fallando, salir
            break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    try:
        detections = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="retinaface",
            enforce_detection=False
        )
    except:
        continue

    for det in detections:
        face_img = det["face"]
        if face_img is None or face_img.size == 0:
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

        person = find_person(embedding)

        if person:
            person["last_seen"] = now
            person["appearances"] += 1

        else:
            pid = person_id
            person_id += 1

            person_dir = os.path.join(OUTPUT_DIR, f"person_{pid}")
            os.makedirs(person_dir, exist_ok=True)

            img_path = os.path.join(person_dir, "face_1.jpg")
            cv2.imwrite(img_path, face_img)

            people.append({
                "id": pid,
                "first_seen": now,
                "last_seen": now,
                "appearances": 1,
                "embedding": embedding,
                "image": img_path
            })
            # guardar inmediatamente al detectar nueva persona
            save_people()

    print(f"👥 Personas detectadas: {len(people)}")

    cap.release()

    # Guardar JSON final
    save_people()

    print("\n✅ Proceso finalizado")
    print(f"📄 Archivo generado: {JSON_FILE}")
