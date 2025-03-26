import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle

# Load model and class names
model = tf.keras.models.load_model(r'C:\Python\Internship\asl_model.h5')
with open(r'C:\Python\Internship\class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)
class_names = {v:k for k,v in class_names.items()}  # Reverse mapping

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def preprocess_hand(frame, landmarks):
    # Extract and expand bounding box
    x_coords = [lm.x * frame.shape[1] for lm in landmarks.landmark]
    y_coords = [lm.y * frame.shape[0] for lm in landmarks.landmark]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    
    expand = 20
    x_min = max(0, x_min - expand)
    x_max = min(frame.shape[1], x_max + expand)
    y_min = max(0, y_min - expand)
    y_max = min(frame.shape[0], y_max + expand)
    
    cropped = frame[y_min:y_max, x_min:x_max]
    resized = cv2.resize(cropped, (200, 200))
    return resized / 255.0

# Webcam loop
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mirror and process
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Preprocess and predict
            processed = preprocess_hand(frame, hand_landmarks)
            prediction = model.predict(np.expand_dims(processed, axis=0))
            class_idx = np.argmax(prediction)
            confidence = np.max(prediction)
            
            if confidence > 0.8:
                letter = class_names[class_idx]
                cv2.putText(frame, f"{letter} ({confidence:.2f})", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('ASL Translator', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()