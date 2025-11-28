# main.py - Run live recognition (hands + face)
import os, time, argparse
import numpy as np
import cv2
import tensorflow as tf
from utils.hand_detector import HandDetector
from utils.face_detector import FaceMeshDetector
from utils.preprocessor import combine_landmarks, normalize_vector
from utils.audio_player import speak

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="model/custom_model.h5", help="Path to keras model")
parser.add_argument("--labels", type=str, default="model/labels.txt", help="Path to labels.txt")
parser.add_argument("--threshold", type=float, default=0.75, help="Confidence threshold")
args = parser.parse_args()

# Load model if exists
model = None
labels = []
if os.path.exists(args.model):
    model = tf.keras.models.load_model(args.model)
    print("[+] Model loaded:", args.model)
else:
    print("[!] Model not found. Run training first (train.py). The app will use no model fallback.")

if os.path.exists(args.labels):
    with open(args.labels, "r", encoding="utf-8") as f:
        labels = [l.strip() for l in f.readlines()]
else:
    labels = ["hola","gracias","no","si","que"]
    print("[!] labels.txt not found; using default labels:", labels)

hd = HandDetector()
fd = FaceMeshDetector()

cap = cv2.VideoCapture(0)
last_speech = 0
debounce = 1.0

print(">>> Starting live recognition. Press ESC to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    hands = hd.get_landmarks(frame)  # list of 21*3 tuples or None
    face = fd.get_landmarks(frame)   # list of 468*3 tuples or None

    combined = combine_landmarks(hands, face)
    display_text = ""

    if combined is not None:
        vec = normalize_vector(combined)
        x = np.expand_dims(vec, axis=0).astype('float32')
        if model is not None:
            preds = model.predict(x, verbose=0)[0]
            idx = int(preds.argmax())
            conf = float(preds[idx])
            if conf >= args.threshold:
                display_text = f"{labels[idx]} ({conf:.2f})"
                # speak with debounce
                if time.time() - last_speech > debounce:
                    speak(labels[idx])
                    last_speech = time.time()
        else:
            display_text = "(no model)"

    # draw info and show
    cv2.putText(frame, display_text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.imshow("LSA Recognizer (Demo)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
