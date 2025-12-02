import numpy as np

def combine_landmarks(hand_vec, face_vec):
    """
    Combina los vectores de mano y rostro en uno solo.
    """
    return np.concatenate([hand_vec, face_vec])

def normalize_vector(vec):
    """
    Normaliza un vector entre 0 y 1
    """
    vec = vec.astype('float32')
    max_val = np.max(np.abs(vec))
    if max_val == 0:
        return vec
    return vec / max_val
