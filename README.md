# UA Money Reader ₴ 📱

A social initiative Android application designed to assist visually impaired individuals in recognizing Ukrainian currency (UAH) using their smartphone camera and audio feedback. 

This project was developed during the Autumn semester at the Data Science Club.

## 🎯 The Problem & Solution
According to the WHO, millions of people have visual impairments. Recognizing money, especially coins and modern banknotes with similar tactile features, is a daily challenge. **UA Money Reader** solves this by leveraging a custom-trained Computer Vision model to detect currency in the frame, calculate the total sum, and announce it via Android's Text-To-Speech (TTS) system.

## ✨ Key Features
* **Custom Object Detection:** Powered by a lightweight **YOLOv8small** model. 
* **Augmented Dataset:** Trained on a custom dataset expanded to ~900 images of Ukrainian bills and coins using various data augmentation techniques via Roboflow.
* **Voice Feedback:** Seamlessly integrates Text-To-Speech to read out the total recognized amount in Ukrainian.
* **Offline Processing:** Inference is done completely on-device using TensorFlow Lite, ensuring fast response times and data privacy.

## 🛠️ Tech Stack
* **Language:** Kotlin
* **UI Framework:** Jetpack Compose
* **Machine Learning:** TensorFlow Lite (TFLite), YOLOv8 architecture
* **Data Engineering:** Roboflow (Dataset generation, annotation, and augmentation)
* **Accessibility:** Android TextToSpeech API

## 🚀 Roadmap & Future Improvements
This project is an evolving MVP. I am currently planning improvements across three main areas:

### 1. Project Structure & Architecture
- [ ] **MVVM Refactoring:** Transition from a "God Object" `MainActivity` to a clean MVVM architecture.
- [ ] **Separation of Concerns:** Extract TFLite initialization, image preprocessing, and NMS (Non-Maximum Suppression) math into a dedicated `ObjectDetector` repository.

### 2. Usability & UX (Accessibility)
- [ ] **In-App Camera (CameraX):** Replace the system camera intent with an integrated, full-screen CameraX feed to eliminate navigation barriers for visually impaired users.
- [ ] **TalkBack Optimization:** Add proper `contentDescription` tags and focus management for seamless screen reader navigation.

### 3. Model Enhancements
- [ ] **Quantization:** Export the model to `int8` or `fp16` to reduce the `.tflite` file size and significantly speed up inference on mobile devices.
- [ ] **Further Dataset Expansion:** Add background/negative images to reduce false positives in complex lighting conditions.

## 👨‍💻 Development
* Gathered, annotated, and augmented the dataset.
* Trained the YOLOv8 model and exported it for mobile use.
* Built the end-to-end Kotlin Android client (TFLite integration, custom NMS logic, UI, and TTS).
