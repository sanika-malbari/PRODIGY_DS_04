import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Set dataset path
data_dir = r'C:\Users\Sanika\OneDrive\Desktop\project 4\leapGestRecog\leapGestRecog'
IMG_SIZE = 64

data = []
labels = []

# Load and preprocess images
for person_dir in os.listdir(data_dir):
    person_path = os.path.join(data_dir, person_dir)
    if not os.path.isdir(person_path):
        continue
    for gesture_dir in os.listdir(person_path):
        gesture_path = os.path.join(person_path, gesture_dir)
        if not os.path.isdir(gesture_path):
            continue
        for img_file in os.listdir(gesture_path):
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(gesture_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print("Unreadable:", img_path)
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(gesture_dir)

# Check if data loaded
print("Loaded images:", len(data))
print("Loaded labels:", len(labels))
print("Detected classes:", sorted(list(set(labels))))

# Convert to numpy arrays
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data, labels_categorical, test_size=0.2, random_state=42
)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

# Compile and train model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model and label encoder
model.save('gesture_model.h5')
np.save('gesture_labels.npy', le.classes_)

print("✅ Model training complete and saved.")
