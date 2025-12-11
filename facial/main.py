import cv2
import onnxruntime as ort
import numpy as np
import time

# Modelos ONNX
det_model = "det_500m.onnx"
attr_model = "genderage.onnx"

# Cargar modelos
det_sess = ort.InferenceSession(det_model, providers=['CPUExecutionProvider'])
attr_sess = ort.InferenceSession(attr_model, providers=['CPUExecutionProvider'])

det_input = det_sess.get_inputs()[0].name
attr_input = attr_sess.get_inputs()[0].name

# Información de forma de entrada del detector
det_input_shape = det_sess.get_inputs()[0].shape
print("Detector input shape (raw):", det_input_shape)
if len(det_input_shape) == 4:
    _, det_c, det_h, det_w = det_input_shape
    # manejar valores None
    det_h = det_h if det_h is not None else 640
    det_w = det_w if det_w is not None else 640
else:
    det_h, det_w = 640, 640

# ---- USAR LA CAMARA WEB ----
cap = cv2.VideoCapture(0)  # 0 = cámara por defecto

# Ajustá la resolución si querés más FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara")
        break

    # ---- PREPROCESAMIENTO PARA SCRFD ----
    img = frame.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    outs = det_sess.run(None, {det_input: img})

    # Depuración: mostrar información de las salidas del modelo
    print("len(outs) =", len(outs))
    for i, o in enumerate(outs):
        print(f"outs[{i}].shape =", np.shape(o))

    # Construir scores y boxes de forma dinámica según las salidas del modelo
    n_outs = len(outs)
    if n_outs == 0:
        scores = np.array([])
        boxes = np.array([]).reshape(0, 4)
    else:
        score_arrays = []
        box_arrays = []
        for o in outs:
            arr = np.array(o)
            # Si la última dimensión es 4 => cajas (x1,y1,x2,y2)
            if arr.ndim >= 2 and arr.shape[-1] == 4:
                box_arrays.append(arr.reshape(-1, 4))
            # Si la última dimensión es 1 => puntuaciones
            elif arr.ndim >= 2 and arr.shape[-1] == 1:
                score_arrays.append(arr.reshape(-1))
            # Si la salida tiene otra forma (p. ej. clases de probabilidad), la ignoramos
            else:
                print(f"Ignorando outs con shape {arr.shape}")

        scores = np.concatenate(score_arrays, axis=0) if len(score_arrays) > 0 else np.array([])
        boxes = np.concatenate(box_arrays, axis=0) if len(box_arrays) > 0 else np.array([]).reshape(0, 4)

    mask = scores > 0.5
    scores = scores[mask]
    boxes = boxes[mask]

    boxes_xywh = []
    for box in boxes:
        x1, y1, x2, y2 = box
        boxes_xywh.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), 0.5, 0.4)
    indices = indices.flatten() if len(indices) > 0 else []

    # ---- PROCESAR CADA ROSTRO ----
    for i in indices:
        x, y, w, h = boxes_xywh[i]

        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue

        # Preprocesamiento AGE/GENDER
        face_resized = cv2.resize(face, (96, 96))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_input = face_rgb.astype(np.float32) / 255.0
        face_input = np.transpose(face_input, (2, 0, 1))
        face_input = np.expand_dims(face_input, axis=0)

        out = attr_sess.run(None, {attr_input: face_input})[0][0]

        gender = "Hombre" if out[0] > out[1] else "Mujer"
        age = int(out[2] * 100)

        label = f"{gender}, {age} años"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # FPS
    now = time.time()
    fps = 1 / (now - prev_time) if prev_time else 0
    prev_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Webcam — Edad y Género", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# from ultralytics import YOLO
# import cv2
# from yt_dlp import YoutubeDL
# import numpy as np
# import insightface
# from insightface.app import FaceAnalysis
# import time

# # ---------- CONFIG -----------
# YOUTUBE_URL = "https://www.youtube.com/watch?v=NK3S_T0Sabk"
# CONF_FACE = 0.5   # confianza mínima detección rostro


# def get_stream_url(youtube_url: str) -> str:
#     ydl_opts = {"format": "best", "quiet": True}
#     with YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(youtube_url, download=False)
#         return info["url"]


# def main():
#     print("Obteniendo stream...")
#     stream = get_stream_url(YOUTUBE_URL)
#     cap = cv2.VideoCapture(stream)

#     if not cap.isOpened():
#         print("❌ No se pudo abrir el stream")
#         return

#     print("Cargando YOLO FACE DETECTOR...")
#     face_model = YOLO("yolo11n-face.pt")

#     print("Cargando modelo InsightFace AGE/GENDER...")
#     app = FaceAnalysis(
#         name="buffalo_l",  # incluye age y gender
#         providers=['CPUExecutionProvider']   # puede usar GPU si tenés
#     )
#     app.prepare(ctx_id=0, det_size=(640, 640))

#     prev_time = 0

#     print("🔥 Sistema listo (detección + sexo + edad)")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # FPS
#         now = time.time()
#         fps = 1 / (now - prev_time) if prev_time else 0
#         prev_time = now

#         # ---- 1) DETECCIÓN DE ROSTROS (YOLO) ----
#         results = face_model(frame, conf=CONF_FACE, verbose=False)

#         annotated = frame.copy()

#         if results and len(results[0].boxes) > 0:
#             for box in results[0].boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])

#                 # recorte del rostro
#                 face_crop = frame[y1:y2, x1:x2]
#                 if face_crop.size == 0:
#                     continue

#                 # ---- 2) AGE + GENDER (InsightFace) ----
#                 faces = app.get(face_crop)

#                 if len(faces) == 0:
#                     continue

#                 f = faces[0]
#                 age = int(f.age)
#                 gender = "HOMBRE" if f.gender == 1 else "MUJER"

#                 # ---- 3) DIBUJAR ----
#                 cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 200), 2)
#                 cv2.putText(
#                     annotated,
#                     f"{gender}, {age} años",
#                     (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (0, 255, 255),
#                     2
#                 )

#         cv2.putText(
#             annotated,
#             f"FPS: {fps:.1f}",
#             (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (255, 255, 255),
#             2,
#         )

#         cv2.imshow("AGE & GENDER", annotated)

#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
