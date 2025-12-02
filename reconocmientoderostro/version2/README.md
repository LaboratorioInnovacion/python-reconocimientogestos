# Sign Language LSTM Recognizer

Proyecto avanzado de reconocimiento de lenguaje de señas (LSA) con:
- Mano + rostro
- Secuencias temporales (LSTM/BiLSTM)
- Audio TTS por gesto reconocido

## Requisitos
pip install opencv-python mediapipe numpy tensorflow pyttsx3 scikit-learn

## Scripts
- `capture_sequence.py` : captura secuencias de gestos desde webcam
- `train_lstm.py` : entrena modelo BiLSTM con secuencias
- `main_lstm.py` : reconoce en vivo y reproduce audio
- `utils/` : detectores, preprocesamiento y TTS
- `dataset/<label>/` : carpetas para cada gesto, contiene `.npy` de secuencias
- `model/` : se guardará `lstm_model.h5` y `labels.txt`

## Flujo
1. Captura secuencias:
   python capture_sequence.py --label hola --seq_len 25 --samples 300
2. Repetir para todas las etiquetas
3. Entrenar:
   python train_lstm.py --dataset dataset --out model/lstm_model.h5
4. Ejecutar en vivo:
   python main_lstm.py --model model/lstm_model.h5
