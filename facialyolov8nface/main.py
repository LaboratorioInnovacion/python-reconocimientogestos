import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import csv
import time
from datetime import datetime

# --------------------------------------------
# CONFIGURACIÓN DE MODELOS
# --------------------------------------------
face_detector = YOLO("models/yolov8n-face.onnx")
genderage_sess = ort.InferenceSession("models/genderage.onnx", providers=['CPUExecutionProvider'])

# --------------------------------------------
# CSV: CREAR ARCHIVO SI NO EXISTE
# --------------------------------------------
csv_file = "detecciones.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "id", "gender", "age", "x1", "y1", "x2", "y2"])


# --------------------------------------------
# FUNCIÓN: CLASIFICAR GÉNERO Y EDAD
# --------------------------------------------
def predict_gender_age(face_img):
    # Preprocesamiento InsightFace: RGB, 96x96, float32, [0,1]
    img_r = cv2.resize(face_img, (96, 96))
    img_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    arr = img_rgb.astype(np.float32) / 255.0
    arr = np.expand_dims(arr.transpose(2, 0, 1), axis=0)
    input_name = genderage_sess.get_inputs()[0].name
    out = genderage_sess.run(None, {input_name: arr})
    vals = out[0][0]
    gender = "Hombre" if vals[0] > vals[1] else "Mujer"
    age = int(max(0, min(vals[2] * 100, 100))) if len(vals) > 2 else -1
    return gender, age


# --------------------------------------------
# CONTADOR + TRACKING SIMPLE
# --------------------------------------------
face_id = 0
last_positions = []

def is_new_face(x, y):
    global last_positions
    for (lx, ly) in last_positions:
        if abs(x - lx) < 50 and abs(y - ly) < 50:
            return False
    last_positions.append((x, y))
    if len(last_positions) > 50:
        last_positions.pop(0)
    return True


# --------------------------------------------
# CAPTURA DE CÁMARA
# --------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo acceder a la webcam")
    exit()

print("📹 Cámara iniciada. Presiona Q para salir.")
prev_time = time.time()


# --------------------------------------------
# LOOP PRINCIPAL
# --------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    results = face_detector(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Nuevo rostro → asignar ID
            if is_new_face(cx, cy):
                face_id += 1

                # Guardar en CSV
                with open(csv_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        face_id,
                        "-", "-",  # luego se completa con predicción
                        x1, y1, x2, y2
                    ])

            # Predicción de género + edad
            gender, age = predict_gender_age(face)

            # Logging en consola
            print(f"[ID {face_id}] {gender}, {age} años – Caja: {x1},{y1},{x2},{y2}")

            # Dibujar rectángulo + texto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, f"{gender}, {age}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    # Mostrar FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

    # Mostrar ventana
    cv2.imshow("YOLO FACE + Edad/Género + FPS", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
