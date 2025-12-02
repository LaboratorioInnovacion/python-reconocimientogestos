import os, argparse, numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="dataset")
parser.add_argument("--out", type=str, default="model/lstm_model.h5")
parser.add_argument("--labels_out", type=str, default="model/labels.txt")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()

def load_sequences(dataset_path):
    X = []
    y = []
    labels = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    for idx, lab in enumerate(labels):
        files = sorted(glob(os.path.join(dataset_path, lab, "*.npy")))
        for f in files:
            arr = np.load(f)
            X.append(arr)
            y.append(idx)
    return np.array(X, dtype='float32'), np.array(y, dtype='int32'), labels

def augment_sequence(seq):
    seq = seq.copy()
    jitter = np.random.normal(0, 0.01, seq.shape).astype('float32')
    seq = seq + jitter
    scale = 1.0 + np.random.normal(0, 0.02)
    seq = seq * scale
    return seq

print("[+] Loading sequences...")
X, y, labels = load_sequences(args.dataset)
print(f"[+] Loaded {X.shape[0]} sequences. Shape per seq: {X.shape[1:]} Labels: {labels}")

min_per_class = 200
aug_X, aug_y = [], []
for label_idx in range(len(labels)):
    idxs = np.where(y == label_idx)[0]
    count = len(idxs)
    if count == 0:
        continue
    for i in idxs:
        aug_X.append(X[i]); aug_y.append(label_idx)
    if count < min_per_class:
        need = min_per_class - count
        for k in range(need):
            src = X[np.random.choice(idxs)]
            aug = augment_sequence(src)
            aug_X.append(aug); aug_y.append(label_idx)

X = np.array(aug_X, dtype='float32')
y = np.array(aug_y, dtype='int32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
cw = {int(c): float(w) for c, w in zip(classes, class_weights)}
print("[+] Class weights:", cw)

seq_len = X_train.shape[1]
feat_dim = X_train.shape[2]

def build_model(seq_len, feat_dim, n_classes):
    inp = tf.keras.layers.Input(shape=(seq_len, feat_dim))
    x = tf.keras.layers.Masking(mask_value=0.0)(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    return model

model = build_model(seq_len, feat_dim, len(labels))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=args.epochs, batch_size=args.batch_size, class_weight=cw, callbacks=[es])

os.makedirs(os.path.dirname(args.out), exist_ok=True)
model.save(args.out)
with open(args.labels_out, 'w', encoding='utf-8') as f:
    for l in labels:
        f.write(l + "\n")
print("[+] Saved model to", args.out)
