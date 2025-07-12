import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
from tensorflow.keras.models import load_model

# Load the trained model and decode class names
model = load_model('gesture_model.h5')
class_names = np.load('gesture_labels.npy')
class_names = [cls.decode('utf-8') if isinstance(cls, bytes) else cls for cls in class_names]

IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.6

# Initialize speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speaking speed

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# For smoothing predictions and avoiding repetition
prediction_history = []
max_history = 5
last_spoken_label = ""

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    label = "No Hand Detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            xmin = int(min(x_coords) * w) - 20
            ymin = int(min(y_coords) * h) - 20
            xmax = int(max(x_coords) * w) + 20
            ymax = int(max(y_coords) * h) + 20

            # Ensure bounds are within frame
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)

            # Extract and preprocess ROI
            roi = frame[ymin:ymax, xmin:xmax]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            img = roi.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict gesture
            predictions = model.predict(img, verbose=0)
            confidence = np.max(predictions)
            class_idx = np.argmax(predictions)

            if confidence > CONFIDENCE_THRESHOLD:
                prediction_history.append(class_idx)
                if len(prediction_history) > max_history:
                    prediction_history.pop(0)
                final_prediction = max(set(prediction_history), key=prediction_history.count)
                gesture_name = class_names[final_prediction]
                label = f"{gesture_name} ({confidence * 100:.1f}%)"

                # Speak only if label changed
                if gesture_name != last_spoken_label:
                    engine.say(f"You showed {gesture_name}")
                    engine.runAndWait()
                    last_spoken_label = gesture_name
            else:
                label = "Uncertain"

            # Draw results
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show prediction
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imshow("Gesture Recognition - MediaPipe + Voice", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
