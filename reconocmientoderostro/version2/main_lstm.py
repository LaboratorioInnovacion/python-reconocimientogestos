import argparse, os, time
import numpy as np
import cv2
from collections import deque
import tensorflow as tf
from utils.hand_detector import HandDetector
from utils.face_detector import FaceMeshDetector
from utils.preprocessor import combine_landmarks, normalize_vector
from utils.audio_player import speak

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="model/lstm_model.h5")
parser.add_argument("--labels", type=str, default="model/labels.txt")
parser.add_argument("--seq_len", type=int, default=25)
parser.add_argument("--threshold", type=float, default=0.70)
parser.add_argument("--smooth_k", type=int, default=5)
args = parser.parse_args()

model = tf.keras.models.load_model(args.model)
with open(args.labels, 'r', encoding='utf-8') as f:
    labels = [l.strip() for l in f.readlines()]

hd = HandDetector()
fd = FaceMeshDetector()
cap = cv2.VideoCapture(0)
buffer = deque(maxlen=args.seq_len)
prob_history = deque(maxlen=args.smooth_k)
last_spoken = 0
debounce = 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hands = hd.get_landmarks(frame)
    face = fd.get_landmarks(frame)
    combined = combine_landmarks(hands, face)
    buffer.append(combined)
    display = ""
    if len(buffer) == args.seq_len and all(v is not None for v in buffer):
        seq = np.array(buffer, dtype='float32')
        seq_norm = normalize_vector(seq.flatten()).reshape(1, args.seq_len, -1).astype('float32')
        preds = model.predict(seq_norm, verbose=0)[0]
        prob_history.append(preds)
        avg = np.mean(np.stack(list(prob_history)), axis=0)
        idx = int(np.argmax(avg))
        conf = float(avg[idx])
        if conf >= args.threshold:
            display = f"{labels[idx]} ({conf:.2f})"
            if time.time() - last_spoken > debounce:
                speak(labels[idx])
                last_spoken = time.time()
        else:
            display = f"({labels[int(np.argmax(avg))]}:{conf:.2f})"

    if display:
        cv2.putText(frame, display, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    cv2.imshow("LSTM Sign Recognizer", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
