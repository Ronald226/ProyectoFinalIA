import tkinter as tk
from tkinter import ttk

import  recognize_speech
from practice_vocabulary import VocabularyPracticeApp
import librosa
import librosa.display
import soundfile as sf
import numpy as np3
import nltk
from nltk.tokenize import word_tokenize
import speech_recognition as sr
import subprocess 

class VoiceAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Language Learning Voice Assistant")

        # Estilo para mejorar el aspecto de la interfaz
        style = ttk.Style()
        style.configure("TButton", padding=10, font=('Helvetica', 12))

        # Agregar título con mejor diseño
        self.title_label = ttk.Label(root, text="Welcome to the Language Learning Voice Assistant", font=('Helvetica', 16))
        self.title_label.pack(pady=20)

        # Agregar imagen (cambia 'your_image.png' al nombre de tu imagen)
        self.image = tk.PhotoImage(file='niño.png')
        self.image_label = ttk.Label(root, image=self.image)
        self.image_label.pack()

        # Crear botones enumerados
        self.buttons = []
        for i, function_name in enumerate(["Recognize Pronunciation", "Practice Speaking", "Practice Vocabulary", "Exit"]):
            button = ttk.Button(root, text=f"{i + 1}. {function_name}", command=lambda fn=function_name: self.execute_function(fn))
            button.pack(pady=10)
            self.buttons.append(button)
        
         # Crear una instancia de PracticeVocabularyWindow
        self.practice_vocabulary_window = None

    def recognize_speech(self):
        with sr.Microphone() as source:
            print("Speak now...")
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio)
                print("You said: " + text)
                return text
            except sr.UnknownValueError:
                print("Sorry, I could not understand what you said.")
                return ""
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
                return ""

    def recognize_pronunciation(self):
        subprocess.run(["python", "recognize_speech.py"])

    def chat_with_ia(self):
        text = self.recognize_speech()
        if text:
            corrected_text = recognize_grammar_errors(text)
            print(f"Corrected text: {corrected_text}")

            

    def practice_vocabulary(self):
            self.practice_vocabulary_window = VocabularyPracticeApp(self.root)
 
  
    def execute_function(self, function_name):
        if function_name == "Recognize Pronunciation":
            self.recognize_pronunciation()
        elif function_name == "Chat with IA":
            self.chat_with_ia()
        elif function_name == "Practice Vocabulary":
            self.practice_vocabulary()
        elif function_name == "Exit":
            self.root.destroy()


if __name__ == "__main__":
    r = sr.Recognizer()
    root = tk.Tk()
    app = VoiceAssistantGUI(root)
    root.mainloop()
