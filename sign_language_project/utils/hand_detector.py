# utils/hand_detector.py
import mediapipe as mp
import numpy as np
import cv2

class HandDetector:
    def __init__(self, max_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=max_hands,
                                         min_detection_confidence=min_detection_confidence,
                                         min_tracking_confidence=min_tracking_confidence)

    def get_landmarks(self, frame):
        # returns flattened list of 21*(x,y,z) in pixel coords normalized by image size
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None
        hand = results.multi_hand_landmarks[0]
        pts = []
        for lm in hand.landmark:
            px = lm.x * w
            py = lm.y * h
            pz = lm.z * max(w,h)
            pts.append((px, py, pz))
        return pts
