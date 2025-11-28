# train.py - trains a simple dense classifier from dataset/<label>/*.npy
import os, argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="dataset", help="Dataset folder containing label subfolders")
parser.add_argument("--out", type=str, default="model/custom_model.h5", help="Output model path")
parser.add_argument("--labels_out", type=str, default="model/labels.txt", help="Labels file")
args = parser.parse_args()

# gather data
X = []
Y = []
labels = sorted([d for d in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, d))])
if not labels:
    raise SystemExit("No label folders found in dataset/")
print("Labels:", labels)
for idx, lab in enumerate(labels):
    folder = os.path.join(args.dataset, lab)
    for f in os.listdir(folder):
        if f.endswith('.npy'):
            arr = np.load(os.path.join(folder, f))
            X.append(arr)
            Y.append(idx)

X = np.array(X, dtype='float32')
Y = np.array(Y, dtype='int32')
print("Samples:", X.shape, "Labels:", len(labels))

# simple shuffle split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42, stratify=Y)

# build model
input_dim = X.shape[1]
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
model.save(args.out)
with open(args.labels_out, 'w', encoding='utf-8') as f:
    for l in labels:
        f.write(l + '\n')
print('[+] Model saved to', args.out)
