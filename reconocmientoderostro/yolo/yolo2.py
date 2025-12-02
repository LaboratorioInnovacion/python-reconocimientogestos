import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from datetime import datetime
from sort import Sort
import mediapipe as mp

# ---------- MODELOS ----------
yolo_people = YOLO("yolov8n.pt")   # detección de personas
tracker = Sort(max_age=15, min_hits=2)

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.45)

# ---------- ESTADOS ----------
processed_ids = set()
count_male = 0
count_female = 0
age_groups = {"0-12":0, "13-20":0, "21-35":0, "36-50":0, "51+":0}

def age_to_group(age):
    if age <= 12: return "0-12"
    if age <= 20: return "13-20"
    if age <= 35: return "21-35"
    if age <= 50: return "36-50"
    return "51+"

# ---------- CAPTURE ----------
cap = cv2.VideoCapture(0)  # o ruta RTSP / archivo
frame_idx = 0
PROCESS_EVERY_N_FRAMES = 2  # ajustar para bajar carga (1 = procesar todo)

print("Iniciando... ESC para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # velocidad: procesar detección cada N frames (pero mostrar siempre)
    do_process = (frame_idx % PROCESS_EVERY_N_FRAMES == 0)

    detections = []
    if do_process:
        # YOLO detecta personas
        results = yolo_people(frame, stream=False)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # persona
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    # recortar dentro frame boundaries
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
                    detections.append([x1, y1, x2, y2, conf])

    # tracker espera np.array((N,5))
    dets_np = np.array(detections) if len(detections)>0 else np.empty((0,5))
    tracks = tracker.update(dets_np)

    # recorrer tracks (x1,y1,x2,y2,track_id)
    for t in tracks:
        x1, y1, x2, y2, track_id = map(int, t)
        # dibujar bbox para feedback visual
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Si no procese esta persona antes, intento detectar cara y analizar
        if track_id not in processed_ids and do_process:
            # recorto region de persona y la convierto a RGB para mediapipe
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

            # MediaPipe face detection en el crop
            res = face_detector.process(rgb_crop)

            if not res.detections:
                # no se detectó cara dentro de la persona
                continue

            # tomamos la detección con mayor score (la 0 suele ser la mejor)
            fd = res.detections[0]
            bboxC = fd.location_data.relative_bounding_box
            h_crop, w_crop = person_crop.shape[:2]

            fx1 = int(bboxC.xmin * w_crop)
            fy1 = int(bboxC.ymin * h_crop)
            fw  = int(bboxC.width * w_crop)
            fh  = int(bboxC.height * h_crop)
            fx2 = fx1 + fw
            fy2 = fy1 + fh

            # chequeos límites
            fx1 = max(0, fx1); fy1 = max(0, fy1)
            fx2 = min(w_crop-1, fx2); fy2 = min(h_crop-1, fy2)

            face_img = person_crop[fy1:fy2, fx1:fx2]
            if face_img.size == 0:
                continue

            try:
                # DeepFace analiza solo la cara recortada
                analysis = DeepFace.analyze(face_img, actions=['age','gender'], enforce_detection=False)
                gender = analysis.get("gender", "")
                age = int(analysis.get("age", 0))
                group = age_to_group(age)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # actualizar contadores
                if gender and "Man" in gender:
                    count_male += 1
                else:
                    count_female += 1

                age_groups[group] += 1
                processed_ids.add(track_id)

                print(f"[{timestamp}] ID={track_id} | {gender} | {age} años | {group}")

            except Exception as e:
                print("DeepFace error:", e)
                # no sumar si falla DeepFace (evita falsas positives)

    # Mostrar contadores en pantalla
    cv2.putText(frame, f"Hombres: {count_male}  |  Mujeres: {count_female}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    y = 60
    for g, v in age_groups.items():
        cv2.putText(frame, f"{g}: {v}", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
        y += 25

    cv2.imshow("Censo - YOLO + MediaPipeFace + DeepFace + SORT", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
