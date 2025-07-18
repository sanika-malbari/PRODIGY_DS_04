# ✋ Real-Time Hand Gesture Recognition with Voice Feedback
> A machine learning project for real-time hand gesture classification using CNNs, MediaPipe, and voice interaction.

![demo](https://user-images.githubusercontent.com/placeholder/demo.gif) <!-- Optional: Add your own GIF or screen recording here -->

## 📌 Project Overview
This project uses a convolutional neural network (CNN) trained on the [LeapGestRecog dataset](https://www.kaggle.com/datasets/kmader/leapgestrecog) to recognize hand gestures in real-time through your webcam. With the help of **MediaPipe**, it dynamically detects your hand and speaks the recognized gesture aloud using **pyttsx3** (text-to-speech).

---

## 🎯 Features

- 🎥 Real-time hand detection using **MediaPipe**
- 🧠 Deep learning model (CNN) trained on LeapGestRecog
- 🗣️ Voice feedback using `pyttsx3`
- 🧽 Smoothing to improve prediction stability
- 🖥️ Works with any standard webcam

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- pyttsx3 (Text-to-Speech)

---

## 🧪 How It Works

1. The webcam captures video frames in real-time.
2. MediaPipe detects the hand and crops the Region of Interest (ROI).
3. The ROI is preprocessed to match training data (grayscale, 64x64).
4. The trained CNN model predicts the gesture class.
5. If confidence is high, the gesture is spoken aloud.

---

## 🚀 Getting Started

### 🔧 Installation

```bash
git clone https://github.com/sanika-malbari/PRODIGY_DS_04.git
cd PRODIGY_DS_04

pip install -r requirements.txt
