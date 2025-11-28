# utils/preprocessor.py
import numpy as np

def combine_landmarks(hands_pts, face_pts):
    # Combine hand (21 points) and face (468 points) landmarks into a single 1D list.
    # If one of them is missing, returns None to avoid noisy samples.
    if hands_pts is None or face_pts is None:
        return None
    # Flatten and concat (hand first, then face)
    hand_flat = [coord for p in hands_pts for coord in p]
    face_flat = [coord for p in face_pts for coord in p]
    combined = hand_flat + face_flat
    return combined

def normalize_vector(vec):
    # Convert to numpy and normalize by mean/std to stabilize values across frames
    arr = np.array(vec, dtype='float32')
    arr = arr - np.mean(arr)
    s = np.std(arr)
    if s < 1e-6:
        s = 1.0
    arr = arr / s
    return arr
