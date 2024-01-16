import os
import pandas as pd
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

import pyaudio
import speech_recognition as sr


def extract_features(file_path, max_length=500):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Ajustar la longitud de los MFCC utilizando padding o truncado
    if mfccs.shape[1] > max_length:
        mfccs = mfccs[:, :max_length]
    else:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Escalar los MFCC
    scaler = StandardScaler()
    mfccs_scaled = scaler.fit_transform(mfccs)

    return mfccs_scaled


# Lee el archivo CSV
csv_path = "cv_corpus_v1/train.csv"
df = pd.read_csv(csv_path)

# Lista para almacenar características y etiquetas
features = []
labels = []

# Recorre cada fila del DataFrame
for index, row in df.iterrows():
    file_path = os.path.join("cv_corpus_v1/train", row['filename'])
    class_label = row['text']

    # Extrae características del audio
    feature = extract_features(file_path)
    features.append(feature)
    labels.append(class_label)

# Convierte las listas a arrays numpy
features = np.array(features)
labels = np.array(labels)

# Asegúrate de que las etiquetas sean numéricas
label_to_index = {label: i for i, label in enumerate(set(labels))}
labels = np.array([label_to_index[label] for label in labels])

# División de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Redimensiona los datos para que se ajusten al modelo LSTM (si es necesario)
if len(X_train.shape) == 2:
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

if len(X_test.shape) == 2:
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Convierte las etiquetas a one-hot encoding
y_train = to_categorical(y_train, num_classes=len(set(labels)))
y_test = to_categorical(y_test, num_classes=len(set(labels)))

# Construcción del modelo de red neuronal
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(set(labels)), activation='softmax'))

# Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluación del modelo
accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy[1]*100:.2f}%")
model.save('modelo002.h5')



