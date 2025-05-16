# Real-Time Spanish Sign Language Recognition

## Project Overview

Real-Time-Spanish-Sign-Language-Recognition is a computer vision-based system designed to translate Spanish Sign Language (LSE) hand gestures into text in real time. The project uses TensorFlow and deep learning techniques to recognize and classify hand gestures, enabling accurate interpretation of static and dynamic signs.

By processing video input from a webcam, the system identifies and interprets hand configurations corresponding to letters or words in the Spanish sign language alphabet. The system employs convolutional neural networks (CNN) and other deep learning architectures to learn the visual patterns and features of different signs, allowing for robust recognition across various lighting conditions and backgrounds.

The system includes a simple graphical user interface that displays the live video feed and the translated text output. This project contributes to improving accessibility for the deaf and hard-of-hearing community and serves as a foundation for more advanced multimodal sign language translation systems.

## Prerequisites

- Python 3.10 or higher
- Webcam (for real-time recognition)
- Git

## Dependencies

Main libraries used in this project:
- TensorFlow 2.x
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib (for visualization)
- Jupyter Notebook

## Datasets

### Spanish Sign Language (SSL) Dictionary Dataset
- Size: ~5GB
- Source: [Spanish Sign Language Dataset](https://www.kaggle.com/datasets/kirlelea/spanish-sign-language-alphabet-static)
- Author: Asadullah Khalid
- License: CC BY-NC-SA 4.0
- Description: A comprehensive collection of Spanish Sign Language gestures covering various words and phrases.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/andrespin0/Real-Time-Spanish-Sign-Language-Recognition.git
   cd Real-Time-Spanish-Sign-Language-Recognition
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and Setup Datasets**
   - Download the SSL Dictionary Dataset from Kaggle
   - Place the downloaded datasets in the following structure:
     ```
     datasets/
     └── SSLdictionary/
     ```

## Project Structure
```
.
├── datasets/               # Dataset storage
│   └── SSLdictionary/     # Spanish Sign Language dataset
├── Entrega2/
│   └── Notebooks/         # Jupyter notebooks for analysis
│       ├── Data-Augmentation.ipynb              # Data augmentation implementation
│       ├── EDA - Dataset augmented.ipynb        # Exploratory analysis of augmented dataset
│       ├── EDA - Normal dataset.ipynb           # Exploratory analysis of original dataset
│       └── Preprocessing-pipeline.ipynb          # Data preprocessing implementation
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Notebooks Description

1. **EDA - Normal dataset.ipynb**
   - Exploratory Data Analysis of the original dataset
   - Analysis of class distribution
   - Image visualization and statistics
   - Dataset quality assessment

2. **Data-Augmentation.ipynb**
   - Implementation of data augmentation techniques
   - Image transformations (rotation, scaling, etc.)
   - Augmented dataset generation
   - Visualization of augmented samples

3. **EDA - Dataset augmented.ipynb**
   - Analysis of the augmented dataset
   - Verification of augmentation results
   - Distribution comparison with original dataset
   - Quality assessment of augmented images

4. **Preprocessing-pipeline.ipynb**
   - Data preprocessing pipeline implementation
   - Image normalization
   - Dataset splitting (train/validation/test)

## Data Processing Pipeline

Our data processing pipeline includes:

1. **Image Preprocessing**
   - Image resizing to (224, 224)
   - Color conversion from BGR to RGB
   - Pixel normalization (scaling to [0,1])
   - Train/validation/test splitting with stratification

2. **Data Augmentation** (using Albumentations library)
   - Geometric transformations:
     - Random rotation (up to 30 degrees)
     - Random horizontal flip
     - Affine transformations (scale and translate)
   - Color/intensity transformations:
     - Random brightness and contrast
     - Gaussian noise
     - Blur effects
   - Advanced transformations:
     - Elastic transformations
     - Grid distortion
     - Optical distortion
   - Background modifications:
     - Coarse dropout

3. **Data Generation**
   - Batch processing with ImageDataGenerator
   - Real-time augmentation during training
   - Stratified data splitting (70/20/10 for train/validation/test)

## Model Architecture

The project uses a deep learning approach with:
- Convolutional Neural Networks (CNN) for feature extraction
- Batch Normalization for training stability
- Dropout layers for preventing overfitting
- Dense layers for final classification

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training/Validation Loss Curves

## Troubleshooting

Common issues and their solutions:

1. **TensorFlow/GPU Issues**
   - Ensure CUDA and cuDNN are properly installed
   - Check TensorFlow-GPU compatibility
   - Verify GPU is recognized by TensorFlow

2. **Memory Issues**
   - Reduce batch size
   - Use data generators
   - Enable memory growth in TensorFlow

3. **Training Issues**
   - Adjust learning rate
   - Modify batch size
   - Check for data imbalance
   - Monitor loss curves
  
## Saved models
[Traditional CNN and Improved CNN (MobileNetV2)](https://drive.google.com/drive/folders/1XXVZWfFf0O9FYYkjPmYTw4ICwZ553vdS?usp=share_link)

## Acknowledgments

- TensorFlow team and community
- Kaggle and dataset contributors
- Spanish Sign Language community for their guidance and feedback

## Bibliography

### Exploratory Data Analysis (EDA)
1. [How to Apply EDA to Different Types of Data](https://www.linkedin.com/advice/3/how-do-you-apply-eda-different-types?lang=es) - LinkedIn Article
2. [EDA for Image Classification](https://medium.com/geekculture/eda-for-image-classification-dcada9f2567a) - Medium Article
3. [EDA: Images Processing and Exploration](https://www.kaggle.com/code/datark1/eda-images-processing-and-exploration) - Kaggle Notebook
4. [Feature Extraction and EDA for Image Classification](https://github.com/henrhoi/image-classification/blob/master/feature_extraction_and_exploratory_data_analysis.ipynb) - GitHub Repository
5. [Data Exploration for Image Segmentation and Object Detection](https://neptune.ai/blog/data-exploration-for-image-segmentation-and-object-detection) - Neptune.ai Blog

### Data Augmentation
1. [Complete Guide to Data Augmentation](https://www.datacamp.com/tutorial/complete-guide-data-augmentation) - DataCamp Tutorial
2. [Image Augmentation using Albumentations](https://medium.com/@mumbaiyachori/image-augmentation-using-albumenation-17a5bf1a874b) - Medium Article
3. [Data Augmentation Tutorial](https://docs.voxel51.com/tutorials/data_augmentation.html) - Voxel51 Documentation
4. [Albumentations with Image Classification Framework](https://stackoverflow.com/questions/71476099/how-to-add-data-augmentation-with-albumentation-to-image-classification-framewor) - Stack Overflow Discussion


* [Andrés Pino](https://github.com/AndresPin0)
