# capture_dataset.py
# Captures combined landmarks (hand + face) and stores them as .npy files per frame.
import os, time, argparse
import numpy as np
import cv2
from utils.hand_detector import HandDetector
from utils.face_detector import FaceMeshDetector
from utils.preprocessor import combine_landmarks

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True, help="Label name (folder created under dataset/)")
parser.add_argument("--samples", type=int, default=200, help="Number of frames to capture")
parser.add_argument("--out", type=str, default="dataset", help="Dataset root folder")
args = parser.parse_args()

out_dir = os.path.join(args.out, args.label)
os.makedirs(out_dir, exist_ok=True)

hd = HandDetector()
fd = FaceMeshDetector()

cap = cv2.VideoCapture(0)
collected = 0
print(f"[+] Starting capture for label={args.label}. Press 'q' to stop early.")

while collected < args.samples:
    ret, frame = cap.read()
    if not ret:
        break
    hands = hd.get_landmarks(frame)
    face = fd.get_landmarks(frame)
    combined = combine_landmarks(hands, face)
    status = f"Collected: {collected}/{args.samples}"
    if combined is not None:
        filename = os.path.join(out_dir, f"{args.label}_{collected:04d}.npy")
        np.save(filename, np.array(combined, dtype='float32'))
        collected += 1
        status += " - saved"
    cv2.putText(frame, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.imshow('Capture', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[+] Capture finished. Saved {collected} samples to {out_dir}")
