import cv2
import numpy as np
import onnxruntime as ort

# ================================
# CONFIG
# ================================
FACE_MODEL = "models/yolov8n-face.onnx"
AGESEX_MODEL = "models/age_gender.onnx"

FACE_CONF = 0.50
FACE_IOU = 0.45
AGESEX_CONF = 0.50

# ================================
# LOAD MODELS
# ================================
face_session = ort.InferenceSession(FACE_MODEL, providers=["CPUExecutionProvider"])
agesex_session = ort.InferenceSession(AGESEX_MODEL, providers=["CPUExecutionProvider"])

# ================================
# YOLO FACE POSTPROCESS
# ================================
def nms(boxes, scores, iou_threshold=0.45):
    idxs = cv2.dnn.NMSBoxes(boxes, scores, FACE_CONF, iou_threshold)
    if len(idxs) == 0:
        return []
    return [i[0] for i in idxs]


def postprocess_yolo_face(output, img_w, img_h):
    detections = []
    boxes = []
    scores = []

    for det in output[0]:
        conf = det[4]
        if conf < FACE_CONF:
            continue

        x, y, w, h = det[0], det[1], det[2], det[3]

        x1 = int((x - w / 2) * img_w)
        y1 = int((y - h / 2) * img_h)
        x2 = int((x + w / 2) * img_w)
        y2 = int((y + h / 2) * img_h)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(conf))

        detections.append([x1, y1, x2, y2, float(conf)])

    if len(detections) == 0:
        return []

    keep = nms(boxes, scores, FACE_IOU)
    return [detections[i] for i in keep]

# ================================
# AGE + GENDER MODEL
# ================================
AGE_BUCKETS = [
    "0-2","4-6","8-12","15-20",
    "21-24","25-32","33-43","44-53",
    "54-63","65+"
]

def predict_age_sex(face_img):
    blob = cv2.resize(face_img, (224, 224))
    blob = blob.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]

    outputs = agesex_session.run(None, {"input": blob})

    gender_logits = outputs[0][0]
    age_logits = outputs[1][0]

    gender = np.argmax(gender_logits)  # 0=female, 1=male
    age = np.argmax(age_logits)

    gender_text = "Hombre" if gender == 1 else "Mujer"
    age_text = AGE_BUCKETS[age]

    score = float(max(gender_logits.max(), age_logits.max()) * 100)

    return gender_text, age_text, score

# ================================
# MAIN LOOP
# ================================
cap = cv2.VideoCapture(0)
print("🔵 Iniciando detección...")

face_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # PREPROCESS YOLO
    img = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blob = img_rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]

    outputs = face_session.run(None, {"images": blob})
    faces = postprocess_yolo_face(outputs, w, h)

    for (x1, y1, x2, y2, conf) in faces:
        face_id += 1

        # recorte
        face_crop = frame[max(0, y1):y2, max(0, x1):x2]

        if face_crop.size == 0:
            continue

        gender, age, score = predict_age_sex(face_crop)

        # LOG REAL LIMPIO
        print(f"[ID {face_id}] {gender} - {age} - Score {score:.2f}")

        # Draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{gender}, {age}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Deteccion Facial YOLO", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
