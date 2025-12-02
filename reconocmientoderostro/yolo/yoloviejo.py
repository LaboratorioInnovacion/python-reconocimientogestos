import cv2
from ultralytics import YOLO
from deepface import DeepFace
from datetime import datetime

# YOLO modelo nano
model = YOLO("yolov8n.pt")  # o "yolov11n.pt" si lo tenés

cap = cv2.VideoCapture(0)

count_male = 0
count_female = 0

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


frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    # Procesamos 1 de cada 3 frames → mejora rendimiento
    if frame_counter % 3 != 0:
        cv2.imshow("YOLO + DeepFace", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # Detectar personas
    results = model(frame, stream=True)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])

            # 0 = persona (para YOLO)
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
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

                    # Actualizo contadores
                    if gender == "Man":
                        count_male += 1
                    else:
                        count_female += 1

                    age_groups[group] += 1

                    print(f"[{timestamp}] {gender} | {age} años | {group}")

                    # Pintar caja y texto
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, f"{gender}, {age}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,(0,255,0), 2)

                except:
                    pass

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

    cv2.imshow("YOLO + DeepFace", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
