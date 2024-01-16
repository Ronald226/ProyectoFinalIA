
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import speech_recognition as sr
import tkinter as tk
from nltk.corpus import wordnet

from tkinter import ttk, messagebox 

class VocabularyPracticeApp:
    def __init__(self, master):
        self.master = master
        self.master.withdraw() 
        self.ventana_cuestionario = tk.Toplevel(master)
        self.ventana_cuestionario.title("Vocabulary Practice")
        
        self.words = []
        self.definitions = []
        self.quiz_words = []
        self.correct_word = ""
        
        # Descargar datos de WordNet
        nltk.download("wordnet")
        
        # Configurar interfaz
        self.label = tk.Label(self.ventana_cuestionario, text="What is the definition of the following word?")
        self.label.pack(pady=10)
        
        self.question_label = tk.Label(self.ventana_cuestionario, text="", font=('Helvetica', 16, 'bold'))
        self.question_label.pack()

        
        self.option_buttons = []
        for i in range(4):
            button = tk.Button(self.ventana_cuestionario, text="", command=lambda i=i: self.check_answer(i + 1))
            button.pack(pady=5)
            self.option_buttons.append(button)
        
        self.next_question_button = tk.Button(self.ventana_cuestionario, text="Next Question", command=self.next_question)
        self.next_question_button.pack(pady=10)
        
        self.return_button = tk.Button(self.ventana_cuestionario, text="Return to Main Menu", command=self.return_to_main_menu)
        self.return_button.pack(pady=10)
        
        # Iniciar el juego
        self.next_question()

    def next_question(self):
        self.words = []
        self.definitions = []
        
        for synset in wordnet.all_synsets():
            if synset.pos() == "n":
                self.words.append(synset.name().split(".")[0])
                self.definitions.append(synset.definition())
                
        self.quiz_words = np.random.choice(self.words, size=4, replace=False)
        self.correct_word = np.random.choice(self.quiz_words)
        
        self.question_label.config(text=self.correct_word)
        
        for i in range(4):
            self.option_buttons[i].config(text=self.definitions[self.words.index(self.quiz_words[i])])

    def check_answer(self, answer):
        if answer == np.where(self.quiz_words == self.correct_word)[0][0] + 1:
            messagebox.showinfo("Correct", "Congratulations! Your answer is correct.")
        else:
            messagebox.showerror("Incorrect", f"Sorry, your answer is incorrect. The correct answer is {np.where(self.quiz_words == self.correct_word)[0][0] + 1}.")

    def return_to_main_menu(self):
        # Llamar a la función de retorno a la interfaz principal
        self.master.deiconify()


if __name__ == "__main__":
    root = tk.Tk()
    app = VocabularyPracticeApp(root)
    root.mainloop()