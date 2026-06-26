

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical

from google.colab import drive
drive.mount('/content/drive')

"""## 2. Load File Paths (00000= Non-Dys, 11111 = Dys)"""

BASE_PATH = "/content/drive/MyDrive"
NON_DYS_PATH = os.path.join(BASE_PATH, "00000")
DYS_PATH = os.path.join(BASE_PATH, "11111")

file_paths = []
labels = []

for f in os.listdir(NON_DYS_PATH):
    if f.endswith(".wav"):
        file_paths.append(os.path.join(NON_DYS_PATH, f))
        labels.append(0)

for f in os.listdir(DYS_PATH):
    if f.endswith(".wav"):
        file_paths.append(os.path.join(DYS_PATH, f))
        labels.append(1)

file_paths = np.array(file_paths)
labels = np.array(labels)

print("Total files:", len(file_paths))

"""## 3. Mel Spectrogram Extraction (Fixed Length)"""

TARGET_SR = 16000
DURATION = 3.0
MAX_SAMPLES = int(TARGET_SR * DURATION)
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64

def extract_mel_fixed(wav_path):
    y, sr = librosa.load(wav_path, sr=TARGET_SR)

    if len(y) < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)))
    else:
        y = y[:MAX_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=y, sr=TARGET_SR,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

"""## 4. Generate Features"""

X = []

for fp in file_paths:
    mel = extract_mel_fixed(fp)
    X.append(mel)

X = np.array(X)
print("Feature shape:", X.shape)

"""## 5. Normalize & Reshape"""

X = (X - np.mean(X)) / (np.std(X) + 1e-8)
X = X[..., np.newaxis]

y_cat = to_categorical(labels, 2)

print(X.shape, y_cat.shape)

"""## 6. Train / Val / Test Split (70/15/15)"""

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_cat, test_size=0.30, random_state=42, stratify=labels
)

y_temp_lbls = np.argmax(y_temp, axis=1)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp_lbls
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

"""# Visualize Melspectogram"""

import matplotlib.pyplot as plt
import librosa.display
import numpy as np

def trim_silence_for_plot(mel, energy_threshold=-60):
    frame_energy = np.mean(mel, axis=0)
    non_silent = np.where(frame_energy > energy_threshold)[0]

    if len(non_silent) == 0:
        return mel[:, :1]

    return mel[:, non_silent[0]: non_silent[-1] + 1]

y_train_lbls = np.argmax(y_train, axis=1)

dysarthria_indices = [i for i, label in enumerate(y_train_lbls) if label == 1]
non_dysarthria_indices = [i for i, label in enumerate(y_train_lbls) if label == 0]

sample_indices = {
    "Non-Dysarthria": non_dysarthria_indices[:2],
    "Dysarthria": dysarthria_indices[:2]
}

plt.figure(figsize=(15, 10))
i = 0

for label_type, indices in sample_indices.items():
    for idx in indices:
        plt.subplot(2, 2, i + 1)

        mel = X_train[idx].squeeze()
        mel = trim_silence_for_plot(mel)

        librosa.display.specshow(
            mel,
            sr=16000,
            x_axis='time',
            y_axis='mel'
        )

        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Mel-Spectrogram ({label_type})")
        plt.tight_layout()
        i += 1

plt.show()

"""## 7. Hybrid CNN + LSTM + GRU Model"""

from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout,
    BatchNormalization, Reshape,
    LSTM, GRU, Dense, Concatenate
)
from tensorflow.keras.models import Model

inputs = Input(shape=(X_train.shape[1], X_train.shape[2], 1))


x = Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
x = MaxPooling2D((2,2))(x)


time_steps = x.shape[2]
features = x.shape[1] * x.shape[3]
x_seq = Reshape((time_steps, features))(x)


lstm_out = LSTM(64)(x_seq)
gru_out  = GRU(64)(x_seq)


fusion = Concatenate()([lstm_out, gru_out])


fc = Dense(64, activation='relu')(fusion)

outputs = Dense(2, activation='softmax')(fc)

hybrid_model = Model(inputs, outputs)

hybrid_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

hybrid_model.summary()



"""## 8. Training"""

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

history = hybrid_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)



"""## 9. Evaluation on Test Set"""

test_loss, test_acc = hybrid_model.evaluate(X_test, y_test, verbose=1)
print("Test Accuracy:", test_acc)

y_pred = np.argmax(hybrid_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_true, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Dysarthria', 'Dysarthria'],
            yticklabels=['Non-Dysarthria', 'Dysarthria'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Hybrid Model')
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_pred_proba = hybrid_model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Hybrid Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
