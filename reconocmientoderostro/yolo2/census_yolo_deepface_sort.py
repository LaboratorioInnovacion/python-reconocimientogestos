import cv2
from ultralytics import YOLO
from deepface import DeepFace
from datetime import datetime
from sort import Sort   # IMPORTANTE
import numpy as np

# ----- CONFIGURACIÓN -----

model = YOLO("yolov8n.pt")  # detección rápida
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

cap = cv2.VideoCapture(0)

# Contadores y registros
count_male = 0
count_female = 0
processed_ids = {}  # Para evitar analizar 2 veces la misma persona

age_groups = {
    "0-12": 0,
    "13-20": 0,
    "21-35": 0,
    "36-50": 0,
    "51+": 0
}

def age_to_group(age):
    if age <= 12: return "0-12"
    if age <= 20: return "13-20"
    if age <= 35: return "21-35"
    if age <= 50: return "36-50"
    return "51+"

print("▶ Iniciando sistema... (ESC para salir)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # persona
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])

    detections = np.array(detections)
    tracks = tracker.update(detections)

    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)

        # Evitar reanalizar la misma persona
        if track_id not in processed_ids:

            face = frame[y1:y2, x1:x2]

            try:
                analysis = DeepFace.analyze(
                    face,
                    actions=['age', 'gender'],
                    enforce_detection=False
                )

                gender = analysis["gender"]
                age = int(analysis["age"])
                group = age_to_group(age)
                timestamp = datetime.now().strftime("%H:%M:%S")

                # Marcar que esta persona ya fue contada
                processed_ids[track_id] = {
                    "gender": gender,
                    "age": age,
                    "group": group,
                    "time": timestamp
                }

                # Actualizo contadores
                if gender == "Man":
                    count_male += 1
                else:
                    count_female += 1

                age_groups[group] += 1

                print(f"[{timestamp}] ID={track_id} | {gender} | {age} años | {group}")

            except:
                pass

        # Dibujar caja + ID
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Mostrar contadores
    cv2.putText(frame, f"Hombres: {count_male}  |  Mujeres: {count_female}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255,255,255), 2)

    y = 60
    for g, total in age_groups.items():
        cv2.putText(frame, f"{g}: {total}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200,200,0), 2)
        y += 25

    cv2.imshow("Censo YOLO + DeepFace + SORT", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
