import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model and label classes
model = load_model('gesture_model.h5')
class_names = np.load('gesture_labels.npy')

IMG_SIZE = 64

# Define Region of Interest (ROI)
x1, y1, x2, y2 = 100, 100, 300, 300  # Larger region for better detection

# Initialize prediction history for smoothing
prediction_history = []
max_history = 5

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Draw the ROI rectangle on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]

    # Preprocess the ROI
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict gesture
    predictions = model.predict(img)
    confidence = np.max(predictions)
    class_idx = np.argmax(predictions)

    if confidence > 0.6:
        prediction_history.append(class_idx)
        if len(prediction_history) > max_history:
            prediction_history.pop(0)

        # Show the most frequent prediction in history
        final_prediction = max(set(prediction_history), key=prediction_history.count)
        label = f"{class_names[final_prediction]} ({confidence * 100:.1f}%)"
    else:
        label = "Uncertain"

    # Display result
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Gesture Recognition", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
