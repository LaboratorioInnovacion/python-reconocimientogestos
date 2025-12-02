import argparse, os, time
import numpy as np
import cv2
from collections import deque
from utils.hand_detector import HandDetector
from utils.face_detector import FaceMeshDetector
from utils.preprocessor import combine_landmarks

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--seq_len", type=int, default=25)
parser.add_argument("--samples", type=int, default=200)
parser.add_argument("--out", type=str, default="dataset")
parser.add_argument("--stride", type=int, default=5)
args = parser.parse_args()

out_dir = os.path.join(args.out, args.label)
os.makedirs(out_dir, exist_ok=True)

hd = HandDetector()
fd = FaceMeshDetector()

cap = cv2.VideoCapture(0)
buffer = deque(maxlen=args.seq_len)
collected = 0
frame_idx = 0

print(f"[+] Start capture for label={args.label}. Press 'q' to stop.")
time.sleep(1.0)

while collected < args.samples:
    ret, frame = cap.read()
    if not ret:
        break

    hands = hd.get_landmarks(frame)
    face = fd.get_landmarks(frame)
    combined = combine_landmarks(hands, face)
    buffer.append(combined)
    frame_idx += 1

    if len(buffer) == args.seq_len and (frame_idx % args.stride == 0):
        if all(f is not None for f in buffer):
            arr = np.array(buffer, dtype='float32')
            filename = os.path.join(out_dir, f"seq_{collected:05d}.npy")
            np.save(filename, arr)
            collected += 1
            print(f"[+] Saved {filename}")

    status = f"Collected: {collected}/{args.samples} | Buffer: {len(buffer)}"
    cv2.putText(frame, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.imshow("Capture Sequences", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[+] Finished capture.")
