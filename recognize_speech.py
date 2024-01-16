import tkinter as tk
from tkinter import Text, Button, Label
import numpy as np
import tensorflow as tf
import librosa
import speech_recognition as sr
import os
from sklearn.preprocessing import StandardScaler
from faker import Faker


# Cargar el modelo LSTM entrenado
model_path = 'mi_modelo_lstm.h5'
model = tf.keras.models.load_model(model_path)

fake = Faker()
recognizer = sr.Recognizer()

def generate_random_text(length=200):
    return fake.text(max_nb_chars=length)
def speech_to_text():
    with sr.Microphone() as source:
        print("Hable algo:")
        audio_data = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio_data, language="en-US")
        return text
    except sr.UnknownValueError:
        print("No se pudo entender el habla.")
        return None
    except sr.RequestError as e:
        print(f"Error en la solicitud a Google API: {e}")
        return None
class SpeechRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Recognition App")

        self.text_to_read = generate_random_text()
        self.label = Label(root, text=f"Texto para leer:\n{self.text_to_read}", wraplength=300)
        self.label.pack()

        self.transcribed_text_box = Text(root, height=5, width=40)
        self.transcribed_text_box.pack()

        self.start_recording_button = Button(root, text="Comenzar Grabación", command=self.record_and_transcribe)
        self.start_recording_button.pack()

    def record_and_transcribe(self):
        self.transcribed_text_box.delete(1.0, tk.END)  # Limpiar el cuadro de texto
        audio_path = "temp_audio.wav"

        with sr.Microphone() as source:
            print("Habla ahora:")
            audio_data = recognizer.listen(source, timeout=5)

        with open(audio_path, "wb") as f:
            f.write(audio_data.get_wav_data())

        predicted_label = predict_pronunciation(audio_path)
        print(f"Predicción de pronunciación: {predicted_label}")

        transcribed_text = speech_to_text()
        if transcribed_text:
            self.transcribed_text_box.insert(tk.END, f"Texto transcrito: {transcribed_text}\n")

        os.remove(audio_path)  # Eliminar el archivo de audio temporal

def extract_features_from_audio(y, sr, max_length=500):
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

def predict_pronunciation(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    features = extract_features_from_audio(y, sr)
    features = features.reshape(1, features.shape[0], features.shape[1])

    prediction = model.predict(features)
    predicted_label = np.argmax(prediction)

    return predicted_label

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechRecognitionApp(root)
    root.mainloop()