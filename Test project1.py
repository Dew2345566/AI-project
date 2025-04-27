import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
from tensorflow.keras.models import load_model
import webbrowser
import tkinter as tk
from tkinter import Button, Label, Frame

# Load the trained model
try:
    model = load_model("best_hand_sign_model.h5", compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load hand tracking model
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5 )

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1.0)  # Volume level

# Dictionary to map predicted labels to words and letters
word_dict = {
    0: "Hello", 1: "Yes", 2: "No", 3: "Thank You", 4: "Please", 5: "Love",
    6: "A", 7: "B", 8: "C", 9: "D", 10: "E", 11: "F", 12: "G", 13: "H", 14: "I", 15: "J", 
    16: "K", 17: "L", 18: "M", 19: "N", 20: "O", 21: "P", 22: "Q", 23: "R", 24: "S", 25: "T", 
    26: "U", 27: "V", 28: "W", 29: "X", 30: "Y", 31: "Z"
}

# Function to process hand sign prediction
def predict_hand_sign(frame, hand_landmarks):
    x_min = min([lm.x for lm in hand_landmarks.landmark])
    y_min = min([lm.y for lm in hand_landmarks.landmark])
    x_max = max([lm.x for lm in hand_landmarks.landmark])
    y_max = max([lm.y for lm in hand_landmarks.landmark])

    h, w, c = frame.shape
    x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)
    
    hand_image = frame[y_min:y_max, x_min:x_max]
    if hand_image.size == 0:
        return "Unknown"
    
    hand_image = cv2.resize(hand_image, (224, 224))
    hand_image = hand_image / 255.0  # Normalize pixel values
    hand_image = np.expand_dims(hand_image, axis=0)  # Add batch dimension

    prediction = model.predict(hand_image)
    predicted_label = np.argmax(prediction)
    return word_dict.get(predicted_label, "Unknown")

# OpenCV for real-time hand tracking
def track_hands():
    cap = cv2.VideoCapture(0)
    sentence = ""
    last_prediction = ""
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                predicted_word = predict_hand_sign(frame, hand_landmarks)
                
                if predicted_word != "Unknown" and predicted_word != last_prediction:
                    sentence += predicted_word + " "
                    last_prediction = predicted_word
                    engine.say(predicted_word)
                    engine.runAndWait()
                
                cv2.putText(frame, f'Prediction: {predicted_word}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Sentence: {sentence}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, sentence, (frame.shape[1]//2 - 100, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Hand Sign Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Function to open webcam
def open_webcam():
    track_hands()

# Function to open web.html
def open_webpage():
    webbrowser.open("web.html")

# Create an improved GUI with better design
def create_gui():
    root = tk.Tk()
    root.title("Hand Sign Tracking")
    root.geometry("500x400")
    root.configure(bg="#e3f2fd")
    
    frame = Frame(root, bg="#ffffff", padx=20, pady=20, relief="ridge", borderwidth=5)
    frame.pack(pady=40)
    
    label = Label(frame, text="Hand Sign Recognition", font=("Arial", 18, "bold"), bg="#ffffff")
    label.pack(pady=15)
    
    btn_webcam = Button(frame, text="Open Webcam", command=open_webcam, font=("Arial", 14), bg="#0288D1", fg="white", padx=10, pady=5, width=20)
    btn_webcam.pack(pady=10)
    
    btn_webpage = Button(frame, text="Go to Web Page", command=open_webpage, font=("Arial", 14), bg="#00796B", fg="white", padx=10, pady=5, width=20)
    btn_webpage.pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    create_gui()
