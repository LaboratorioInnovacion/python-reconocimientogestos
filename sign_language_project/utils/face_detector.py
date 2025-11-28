# utils/face_detector.py
import mediapipe as mp
import numpy as np
import cv2

class FaceMeshDetector:
    def __init__(self, min_detection_confidence=0.6):
        self.mp_face = mp.solutions.face_mesh
        self.face = self.mp_face.FaceMesh(static_image_mode=False,
                                         max_num_faces=1,
                                         refine_landmarks=False,
                                         min_detection_confidence=min_detection_confidence,
                                         min_tracking_confidence=0.6)

    def get_landmarks(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face.process(rgb)
        if not results.multi_face_landmarks:
            return None
        face = results.multi_face_landmarks[0]
        pts = []
        for lm in face.landmark:
            px = lm.x * w
            py = lm.y * h
            pz = lm.z * max(w,h)
            pts.append((px, py, pz))
        return pts
