import tkinter as tk
from tkinter import Scrollbar, Text, Entry, Button
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot GUI")

        # Create and configure the text widget
        self.chat_display = Text(root, wrap="word", width=40, height=10, state="disabled")
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Create a scrollbar for the text widget
        scrollbar = Scrollbar(root, command=self.chat_display.yview)
        scrollbar.grid(row=0, column=2, sticky="nsew")
        self.chat_display['yscrollcommand'] = scrollbar.set

        # Create and configure the entry widget for user input
        self.user_input_entry = Entry(root, width=30)
        self.user_input_entry.grid(row=1, column=0, padx=10, pady=10)

        # Create a button for sending user input
        send_button = Button(root, text="Send", command=self.send_user_input)
        send_button.grid(row=1, column=1, pady=10)

        # Initialize the chatbot
        self.initialize_chatbot()

    def initialize_chatbot(self):
        self.conversations = [
            {"input": "Hello", "output": "Hi there!"},
            {"input": "How are you?", "output": "I'm good, thank you."},
            {"input": "Hello", "output": "Hi there!"},
            {"input": "How are you?", "output": "I'm good, thank you."},
            {"input": "What's your name?", "output": "I don't have a name."},
            {"input": "Tell me a joke", "output": "Why don't scientists trust atoms? Because they make up everything."},
            {"input": "How is the weather today?", "output": "I'm not sure. I'm just a chatbot."},
            {"input": "What is the meaning of life?", "output": "The meaning of life is a philosophical question."},
        ]
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit([conv["input"] for conv in self.conversations])  # Ajustar el vectorizador con los datos iniciales

        # Crear el clasificador y ajustarlo
        corpus = [conv["input"] for conv in self.conversations]
        X = self.vectorizer.transform(corpus).toarray()
        y = np.array(range(len(self.conversations)))
        self.label_to_response = {i: conv["output"] for i, conv in enumerate(self.conversations)}
        self.classifier = MultinomialNB()
        self.classifier.fit(X, y)

        self.update_chat_display("Chatbot: ¡Hola! Escribe 'exit' para finalizar la conversación.")
        

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)

    def get_response(self, user_input):
        user_input = self.preprocess_text(user_input)
        input_vector = self.vectorizer.transform([user_input]).toarray()
        predicted_label = self.classifier.predict(input_vector)[0]
        response = self.label_to_response.get(predicted_label, "Lo siento, no entiendo.")
        return response

    def update_chat_display(self, message):
        self.chat_display.config(state="normal")
        self.chat_display.insert(tk.END, message + "\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state="disabled")

    def send_user_input(self):
        user_input = self.user_input_entry.get()
        if user_input.lower() == 'exit':
            self.update_chat_display("Chatbot: ¡Adiós!")
            self.root.quit()
        else:
            response = self.get_response(user_input)
            self.update_chat_display(f"Tú: {user_input}")
            self.update_chat_display(f"Chatbot: {response}")
            self.user_input_entry.delete(0, tk.END)

            # Actualizar los datos de la conversación
            new_conversation = {"input": user_input, "output": response}
            new_conversation["input"] = self.preprocess_text(new_conversation["input"])
            new_conversation["output"] = self.preprocess_text(new_conversation["output"])
            self.conversations.append(new_conversation)

            # Actualizar el vectorizador y el clasificador
            corpus = [conv["input"] for conv in self.conversations]
            X = self.vectorizer.transform(corpus).toarray()
            y = np.array(range(len(self.conversations)))
            self.label_to_response = {i: conv["output"] for i, conv in enumerate(self.conversations)}
            self.classifier = MultinomialNB()
            self.classifier.fit(X, y)
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
