import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    try:
        # Analizar cada N frames para mejor rendimiento
        analysis = DeepFace.analyze(
            frame, 
            actions=['age', 'gender'],
            enforce_detection=False,
            detector_backend='opencv'  # Más rápido en CPU
        )
        
        # Extraer información
        age = analysis[0]['age']
        gender = analysis[0]['dominant_gender']
        
        # Mostrar en pantalla
        cv2.putText(frame, f"Edad: {age}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Genero: {gender}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    except:
        pass
    
    cv2.imshow('Deteccion Facial', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()