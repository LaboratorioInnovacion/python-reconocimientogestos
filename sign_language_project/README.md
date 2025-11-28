# Sign Language Recognizer - Base Project (LSA: 5 words)
This project detects hands + face landmarks (MediaPipe) and allows you to:
- Capture dataset frames (combined hand + face landmarks) for 5 default words:
  `hola`, `gracias`, `no`, `si`, `que`
- Train a Keras model on the captured `.npy` landmark vectors
- Run live recognition with TTS voice output for recognized words

## Requirements
pip install opencv-python mediapipe numpy tensorflow pyttsx3 scikit-learn

## Folders
- `dataset/<label>/` - contains `.npy` files with combined landmarks
- `model/` - saved keras model (`custom_model.h5`) and `labels.txt`

## Typical workflow
1. Capture samples:
   python capture_dataset.py --label hola --samples 200
2. Repeat for other labels: gracias, no, si, que
3. Train:
   python train.py --dataset dataset --out model/custom_model.h5
4. Predict live:
   python main.py --model model/custom_model.h5
