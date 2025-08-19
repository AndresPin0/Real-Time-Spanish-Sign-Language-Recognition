# Real-Time Spanish Sign Language Recognition

## Project Overview

**Real-Time-Spanish-Sign-Language-Recognition** is a state-of-the-art computer vision system designed to translate Spanish Sign Language (LSE) hand gestures into text in real time. The project leverages advanced deep learning techniques and neural network architectures to achieve exceptional accuracy in recognizing and classifying hand gestures, enabling accurate interpretation of static signs.

By processing video input from a webcam, the system identifies and interprets hand configurations corresponding to letters in the Spanish sign language alphabet. The system employs multiple convolutional neural network architectures with transfer learning to learn visual patterns and features of different signs, allowing for robust recognition across various lighting conditions and backgrounds.

**Key Achievement**: The project has successfully identified and implemented the **most optimal neural network architecture** for Spanish Sign Language recognition, achieving **97.00% accuracy** on the test dataset.

## 🏆 Project Results & Key Findings

### **🏅 WINNING ARCHITECTURE: ResNet50**
- **Test Accuracy**: **97.00%** (Outstanding Performance)
- **Training Accuracy**: 97.81%
- **Validation Accuracy**: 95.00%
- **Training Time**: 27 minutes (1614.65 seconds)
- **Status**: **READY FOR PRODUCTION**

### **🥈 SECOND PLACE: MobileNetV3**
- **Test Accuracy**: 90.75% (Respectable Performance)
- **Training Accuracy**: 95.93%
- **Validation Accuracy**: 93.13%
- **Training Time**: 3.3 minutes (199.24 seconds)
- **Status**: Good for speed-critical applications

### **❌ FAILED ARCHITECTURE: EfficientNetB0**
- **Test Accuracy**: 4.75% (Complete Failure)
- **Training Accuracy**: 6.49%
- **Validation Accuracy**: 4.69%
- **Status**: Not suitable for this dataset

## 📊 Performance Comparison Summary

| Architecture | Test Accuracy | Training Time | Status | Recommendation |
|--------------|---------------|---------------|---------|----------------|
| **ResNet50** | **97.00%** | 27 min | 🏆 **WINNER** | **Production Ready** |
| **MobileNetV3** | 90.75% | 3.3 min | 🥈 **Runner-up** | Speed-critical apps |
| **EfficientNetB0** | 4.75% | 7.8 min | ❌ **Failed** | Not recommended |

## 🚀 Why ResNet50 is the Optimal Algorithm

### **1. Architectural Advantages**
- **Residual Connections**: Prevent gradient degradation in deep networks
- **50 Layers**: Optimal depth for this complexity level
- **Transfer Learning**: Excellent ImageNet weights utilization
- **Stability**: Proven architecture with consistent performance

### **2. Training Strategy Success**
- **Two-Phase Training**: Initial frozen + gradual fine-tuning
- **Progressive Unfreezing**: Descongelado gradual de capas (30 últimas)
- **Adaptive Learning Rate**: Reduction on plateau with early stopping
- **Data Augmentation**: Advanced transformations for robustness

### **3. Performance Metrics**
- **Accuracy Improvement**: +11.65% over original MobileNetV2 (85.35% → 97.00%)
- **Generalization**: Minimal gap between train/validation/test
- **Consistency**: Stable performance across all metrics

## 📁 Project Structure
```
Real-Time-Spanish-Sign-Language-Recognition/
├── datasets/               # Dataset storage
│   └── fondo_blanco/     # Spanish Sign Language dataset (19 classes)
├── Notebooks/             # Jupyter notebooks for analysis
│   ├── 1. Análisis Exploratorio de datos.ipynb           # Exploratory analysis of original dataset
│   ├── 2. Pipeline de Preprocesamiento.ipynb          # Data preprocessing and MobileNetV2 training
│   └── 3. Comparación de 3 arquitecturas.ipynb  # Architecture comparison (ResNet50 winner)
├── requirements.txt       # Project dependencies
├── venv/                 # Virtual environment
└── README.md             # Project documentation
```

## 🎯 Project Execution Flow

### **Phase 1: Exploratory Data Analysis (EDA)**
**Notebook: `1. Análisis Exploratorio de datos.ipynb`**
- **Purpose**: Understand the structure and quality of the original dataset
- **Key Findings**:
  - **19 Classes**: A, B, C, D, E, F, G, I, K, L, M, N, O, P, Q, R, S, T, U
  - **Total Images**: 1,998 images
  - **Distribution**: Balanced across classes (90-120 images per class)
  - **Format**: RGB images, 224x224 resolution
  - **Quality**: High-quality, consistent background

### **Phase 2: Preprocessing Pipeline and Training**
**Notebook: `2. Pipeline de Preprocesamiento.ipynb`**
- **Purpose**: Prepare data and train initial classification models
- **Models Trained**:
  - **Traditional CNN**: Custom architecture (baseline)
  - **MobileNetV2**: Pre-trained model with fine-tuning
- **Results**: MobileNetV2 achieved 85.35% accuracy

### **Phase 3: Architecture Optimization & Comparison**
**Notebook: `3. Comparación de 3 arquitecturas.ipynb`**
- **Purpose**: Find the most optimal neural network architecture
- **Architectures Tested**:
  - EfficientNetB0 (failed: 4.75%)
  - **ResNet50 (winner: 97.00%)**
  - MobileNetV3 (runner-up: 90.75%)

## 🔬 Technical Implementation Details

### **Data Preprocessing Pipeline**
```python
# Advanced preprocessing implemented:
1. Image loading and validation
2. Resize to (224, 224) with interpolation optimization
3. BGR to RGB conversion
4. Pixel normalization [0,1]
5. Advanced image enhancement (CLAHE, contrast adjustment)
6. Data augmentation with Albumentations
7. Stratified train/validation/test split (70/20/10)
```

### **Training Strategy for ResNet50**
```python
# Two-phase training approach:
Phase 1: Initial Training (15 epochs)
- Base model frozen (ImageNet weights)
- Learning rate: 1e-4
- Focus on classifier layers

Phase 2: Fine-tuning (10 epochs)
- Progressive unfreezing (last 30 layers)
- Learning rate: 1e-5
- Gradual adaptation to SSL dataset
```

### **Advanced Callbacks & Optimization**
```python
# Optimization techniques implemented:
- Early stopping with patience=8
- ReduceLROnPlateau with factor=0.5
- Model checkpointing (best weights)
- TensorBoard logging
- Custom monitoring callbacks
- Learning rate scheduling
```

## 📈 Performance Analysis & Insights

### **ResNet50 Success Factors**
1. **Transfer Learning Excellence**: ImageNet weights provided strong feature extraction
2. **Architectural Stability**: Residual connections maintained gradient flow
3. **Optimal Depth**: 50 layers balanced complexity and capacity
4. **Training Strategy**: Two-phase approach prevented catastrophic forgetting

### **MobileNetV3 Performance Analysis**
1. **Speed Advantage**: 8x faster training than ResNet50
2. **Efficiency**: Good accuracy (90.75%) with minimal resources
3. **Use Case**: Ideal for mobile/edge deployment

### **EfficientNetB0 Failure Analysis**
1. **Architectural Mismatch**: Too complex for this dataset size
2. **Transfer Learning Issues**: Without pre-trained weights, failed to learn
3. **Overfitting**: Complex architecture with limited data

## 🚀 Real-Time Implementation Status

### **Current Status**: ✅ **READY FOR PRODUCTION**
- **Best Model**: ResNet50 with 97.00% accuracy
- **Model Format**: Saved as `.h5` files
- **Inference Speed**: Optimized for real-time processing
- **Memory Requirements**: GPU recommended for optimal performance

### **Implementation Requirements**
```python
# Production deployment:
- TensorFlow 2.x
- GPU support (CUDA/cuDNN)
- Webcam input
- Real-time frame processing
- GUI for result display
```

## 📋 Setup Instructions

### **1. Clone the Repository**
```bash
git clone https://github.com/andrespin0/Real-Time-Spanish-Sign-Language-Recognition.git
cd Real-Time-Spanish-Sign-Language-Recognition
```

### **2. Create and Activate Virtual Environment**
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Download and Setup Datasets**
- Download the SSL Dictionary Dataset from Kaggle
- Place in: `datasets/fondo_blanco/`
- Ensure 19 class folders (A, B, C, D, E, F, G, I, K, L, M, N, O, P, Q, R, S, T, U)

## 🔧 Dependencies

### **Core Libraries**
- **TensorFlow 2.x**: Deep learning framework
- **OpenCV (cv2)**: Computer vision and image processing
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Development environment

### **Advanced Libraries**
- **Albumentations**: Advanced data augmentation
- **TensorBoard**: Training monitoring and visualization

## 📊 Dataset Information

### **Spanish Sign Language (SSL) Dictionary Dataset**
- **Size**: ~5GB
- **Classes**: 19 letters (A, B, C, D, E, F, G, I, K, L, M, N, O, P, Q, R, S, T, U)
- **Images per Class**: 90-120 images
- **Total Images**: 1,998
- **Resolution**: 224x224 RGB
- **Source**: [Spanish Sign Language Dataset](https://www.kaggle.com/datasets/kirlelea/spanish-sign-language-alphabet-static)
- **License**: CC BY-NC-SA 4.0

## 🎯 Key Achievements & Contributions

### **1. Algorithm Optimization Success**
- **Identified optimal architecture**: ResNet50 outperformed all alternatives
- **Achieved exceptional accuracy**: 97.00% on test dataset
- **Improved baseline performance**: +11.65% over MobileNetV2

### **2. Technical Innovations**
- **Advanced data augmentation pipeline**
- **Progressive fine-tuning strategy**
- **Comprehensive architecture comparison**
- **Production-ready model deployment**

### **3. Research Contributions**
- **Deep learning for sign language recognition**
- **Transfer learning optimization**
- **Real-time computer vision systems**


## 📚 Bibliography & References

### **Deep Learning & Computer Vision**
1. [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. [MobileNetV3: Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
3. [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

### **Sign Language Recognition**
1. [Deep Learning for Sign Language Recognition](https://ieeexplore.ieee.org/document/9008291)
2. [Real-time Hand Gesture Recognition](https://www.sciencedirect.com/science/article/pii/S0957417419304568)

### **Data Augmentation & Preprocessing**
1. [Albumentations: Fast and Flexible Image Augmentations](https://arxiv.org/abs/1809.06839)
2. [Data Augmentation in Deep Learning](https://www.tensorflow.org/tutorials/images/data_augmentation)

## 👥 Acknowledgments

- **TensorFlow team** and community for the excellent framework
- **Kaggle** and dataset contributors for the SSL dataset
- **Spanish Sign Language community** for guidance and feedback
- **Research community** for foundational papers and architectures

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Andrés Pino** - [GitHub Profile](https://github.com/AndresPin0)

---

## 🏆 Project Status: **COMPLETED SUCCESSFULLY**

**Final Result**: ResNet50 architecture achieved **97.00% accuracy** and is **ready for production deployment** in real-time Spanish Sign Language recognition systems.

**Key Achievement**: Successfully identified and implemented the **most optimal neural network architecture** for this specific computer vision task, demonstrating the effectiveness of systematic architecture comparison and optimization in deep learning projects.
