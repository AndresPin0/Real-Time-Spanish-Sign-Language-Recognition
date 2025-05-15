# Real-Time Spanish Sign Language Recognition

## Datasets

This project uses two main datasets:

### Spanish Sign Language (SSL) Dictionary Dataset
- Size: ~5GB
- Source: [Spanish Sign Language Dataset](https://www.kaggle.com/datasets/asadullah92/spanish-sign-language-dataset)
- Author: Asadullah Khalid
- License: CC BY-NC-SA 4.0
- Description: A comprehensive collection of Spanish Sign Language gestures covering various words and phrases.

### Mexican Sign Language (MSL) Words Dataset
- Size: ~2GB
- Source: [Mexican Sign Language Dataset](https://www.kaggle.com/datasets/eloyrj/mexican-sign-language-dataset)
- Author: Eloy Rojo
- License: CC0: Public Domain
- Description: A collection of Mexican Sign Language gestures focusing on common words.

## Dataset Access

Due to the large size of the datasets, they are not included directly in this repository. Instead, you can:

1. Download them manually from the Kaggle links above
2. Use the automated download script provided in `download_datasets.py`

Place the downloaded datasets in the following structure:
```
datasets/
├── SSLdictionary/
└── MSLwords1/
```

Real-Time-Spanish-Sign-Language-Recognition is a computer vision-based system designed to translate Spanish Sign Language (LSE) hand gestures into text in real time. The project uses MediaPipe Hands to extract 21 key landmarks per hand, enabling the accurate tracking of static and dynamic gestures.

By processing video input from a webcam, the system identifies and interprets hand configurations corresponding to letters or words in the Spanish sign language alphabet. Feature engineering techniques are applied to extract distances, angles, and motion patterns from the hand keypoints. These features are then fed into machine learning classifiers (e.g., SVM, Random Forest, LSTM) trained to recognize specific signs.

The system includes a simple graphical user interface that displays the live video feed, visualized hand keypoints, and the translated text output. This project contributes to improving accessibility for the deaf and hard-of-hearing community and serves as a foundation for more advanced multimodal sign language translation systems.
