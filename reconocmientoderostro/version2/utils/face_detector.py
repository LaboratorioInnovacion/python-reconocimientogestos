import cv2
import mediapipe as mp
import numpy as np

class FaceMeshDetector:
    def __init__(self, max_faces=1, detection_conf=0.7, tracking_conf=0.7):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf)
        self.mp_draw = mp.solutions.drawing_utils

    def get_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            self.mp_draw.draw_landmarks(frame, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION)
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            return np.array(landmarks).flatten()
        else:
            return np.zeros(468*3)  # 468 puntos de FaceMesh
