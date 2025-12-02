import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, max_hands=2, detection_conf=0.7, tracking_conf=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf)
        self.mp_draw = mp.solutions.drawing_utils

    def get_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            all_hands = []
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                all_hands.append(np.array(landmarks).flatten())
            # Concatenar si hay más de una mano
            return np.concatenate(all_hands)
        else:
            # Si no detecta manos, devolver vector nulo
            return np.zeros(21*3*2)  # máximo 2 manos
