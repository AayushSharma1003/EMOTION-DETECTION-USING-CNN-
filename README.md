#  Real-Time Emotion Detection

A deep learning-based real-time facial emotion detection system that uses a custom Convolutional Neural Network (CNN) trained on the FER-2013 dataset to classify human emotions through a live webcam feed.

---

##  Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Emotion Classes](#emotion-classes)
- [Limitations](#limitations)

---

##  Overview

This project implements a real-time facial emotion recognition system. It detects faces in a live webcam stream using OpenCV's Haar Cascade classifier and then predicts the emotional state of each detected face using a trained deep CNN model.

The system can recognize **7 emotions** in real time:
`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

---

##  Dataset

**FER-2013 (Facial Expression Recognition 2013)**

- **Source:** [Kaggle - FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- **Image size:** 48×48 pixels, grayscale
- **Total training samples:** 28,821 images
- **Total test samples:** 7,066 images
- **Classes:** 7 emotion categories

| Split | Samples |
|-------|---------|
| Train | 28,821  |
| Test  | 7,066   |

---

##  Model Architecture

A custom Sequential CNN built with TensorFlow/Keras:

```
Input: (48, 48, 1)
│
├── Conv2D(128, 3×3, ReLU)
├── MaxPooling2D(2×2)
├── Dropout(0.4)
│
├── Conv2D(256, 3×3, ReLU)
├── MaxPooling2D(2×2)
├── Dropout(0.4)
│
├── Conv2D(512, 3×3, ReLU)
├── MaxPooling2D(2×2)
├── Dropout(0.4)
│
├── Conv2D(512, 3×3, ReLU)
├── MaxPooling2D(2×2)
├── Dropout(0.4)
│
├── Flatten()
│
├── Dense(512, ReLU)
├── Dropout(0.4)
├── Dense(256, ReLU)
├── Dropout(0.3)
│
└── Dense(7, Softmax)  →  Output
```

- **Total convolutional blocks:** 4
- **Fully connected layers:** 2 (512 → 256)
- **Output layer:** 7 neurons with Softmax activation
- **Dropout regularization** applied throughout to prevent overfitting

---

##  Training Details

| Parameter        | Value                     |
|-----------------|---------------------------|
| Optimizer        | Adam                      |
| Loss Function    | Categorical Cross-Entropy |
| Batch Size       | 128                       |
| Epochs           | 50                        |
| Validation Split | Test set (7,066 images)   |
| Input Shape      | (48, 48, 1) — Grayscale   |
| Normalization    | Pixel values ÷ 255.0      |
| Label Encoding   | One-hot encoding (7 classes) |

---

##  Performance

| Metric              | Score   |
|--------------------|---------|
| **Training Accuracy**  | ~63.7%  |
| **Validation Accuracy**| ~62.1%  |
| **Training Loss**      | ~0.959  |
| **Validation Loss**    | ~1.026  |

> Results recorded at **Epoch 50/50**. The model shows a steady improvement over 50 epochs, with validation accuracy rising from ~26% (Epoch 1) to ~62% (Epoch 50).

---

##  Project Structure

```
emotion-detection/
│
├── images/
│   ├── train/          # Training images (organized by emotion label)
│   └── test/           # Test images (organized by emotion label)
│
├── emotiondetector.h5          # Saved Keras model weights
├── emotiondetector.json        # Saved model architecture (JSON)
│
├── Untitled1.ipynb             # Model training notebook
├── realtimedetection.py        # Real-time webcam emotion detection script
│
└── README.md
```

---

##  Requirements

- Python 3.10+
- TensorFlow / Keras
- OpenCV (`cv2`)
- NumPy
- scikit-learn
- Pillow
- tqdm
- pandas
- matplotlib

Install all dependencies:

```bash
pip install tensorflow keras opencv-python numpy scikit-learn pillow tqdm pandas matplotlib keras_preprocessing
```

---

##  Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Download the FER-2013 dataset** from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place images in the `images/train/` and `images/test/` directories.

4. **Train the model** (optional — skip if using the pre-trained `.h5` file):

```bash
jupyter notebook Untitled1.ipynb
```

---

##  Usage

Run real-time emotion detection using your webcam:

```bash
python realtimedetection.py
```

- A window labeled **"Output"** will appear showing the webcam feed with detected faces highlighted by blue rectangles and emotion labels displayed above them.
- Press **`ESC`** to exit the application.

---

##  How It Works

1. **Face Detection:** OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`) detects faces in each webcam frame.
2. **Preprocessing:** Each detected face is:
   - Cropped from the frame
   - Converted to grayscale
   - Resized to 48×48 pixels
   - Normalized (pixel values divided by 255)
3. **Prediction:** The preprocessed face is fed into the trained CNN model.
4. **Display:** The predicted emotion label is overlaid on the original frame above the detected face bounding box.

---

##  Emotion Classes

| Label     | Index |
|-----------|-------|
| Angry     | 0     |
| Disgust   | 1     |
| Fear      | 2     |
| Happy     | 3     |
| Neutral   | 4     |
| Sad       | 5     |
| Surprise  | 6     |

---

##  Limitations

- The FER-2013 dataset is known for noise and label ambiguity, especially for emotions like **fear** and **disgust**, which are harder to distinguish visually.
- The model may struggle in low-light conditions or with partially occluded faces.
- Haar Cascade face detection may miss faces at extreme angles.
- Validation accuracy of ~62% reflects the inherent difficulty of emotion classification from grayscale, low-resolution images.

---

##  License

This project is open-source and available under the [MIT License](LICENSE).

---

## Acknowledgements

- [FER-2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) by msambare
- OpenCV for face detection utilities
- TensorFlow/Keras for the deep learning framework
