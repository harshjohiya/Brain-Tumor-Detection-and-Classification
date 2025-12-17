<div align="center">

# 🧠 Brain Tumor Detection & Classification

### Deep Learning-Powered MRI Analysis System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

*An advanced Convolutional Neural Network (CNN) for automated brain tumor detection and classification from MRI scans*

</div>

---

## 📋 Overview

This project leverages deep learning to classify brain MRI scans into **four distinct categories** with high accuracy:

- 🔴 **Glioma Tumor** - Malignant brain tumor from glial cells
- 🟠 **Meningioma Tumor** - Tumor arising from meninges
- 🟡 **Pituitary Tumor** - Tumor in the pituitary gland
- 🟢 **No Tumor** - Healthy brain scan

Built with **Keras/TensorFlow**, this CNN model provides a robust solution for medical image classification.

---

## 🎯 Key Features

✨ **High Accuracy Classification** - Multi-class tumor detection  
🚀 **Easy to Deploy** - Simple setup with Jupyter Notebook  
📊 **Visual Results** - Clear prediction visualizations  
🔄 **Data Augmentation** - Improved generalization with augmented training  
💾 **Pre-organized Dataset** - Ready-to-use training and testing data  

---

## 📁 Dataset Structure

The dataset is **pre-organized** and included in this repository:

```
📦 Dataset/
 ┣ 📂 Training/
 ┃ ┣ 📁 glioma_tumor/
 ┃ ┣ 📁 meningioma_tumor/
 ┃ ┣ 📁 no_tumor/
 ┃ ┗ 📁 pituitary_tumor/
 ┗ 📂 Testing/
   ┣ 📁 glioma_tumor/
   ┣ 📁 meningioma_tumor/
   ┣ 📁 no_tumor/
   ┗ 📁 pituitary_tumor/
```

> **⚠️ Important Note:** The notebook references a Kaggle path. For local execution, update these variables:
> ```python
> trainPath = "../Dataset/Training"
> testPath = "../Dataset/Testing"
> ```

---

## 🏗️ Model Architecture

Our CNN model features a **carefully designed architecture** for optimal performance:

| Component | Specification |
|-----------|---------------|
| 📐 Input Size | 150×150×3 (RGB) |
| 🔲 Layers | Conv2D + ReLU + MaxPool2D blocks |
| 🎲 Regularization | Dropout layers |
| 🧮 Dense Layers | Dense(1024, ReLU) → Dense(4, Softmax) |
| ⚙️ Optimizer | Adam (lr=0.001) |
| 📉 Loss Function | Categorical Crossentropy |
| 🔄 Training | 40 epochs, batch size 40 |
| 🖼️ Augmentation | Horizontal flip |

📓 **Notebook Location:** [Model/brain_tumor_detection_and_classification.ipynb](Model/brain_tumor_detection_and_classification.ipynb)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

**1️⃣ Clone the repository**
```bash
git clone <your-repo-url>
cd Brain-Tumor-Detection-and-Classification
```

**2️⃣ Create a virtual environment**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3️⃣ Install dependencies**
```powershell
pip install tensorflow numpy pillow scikit-learn matplotlib pandas jupyter
```

**4️⃣ Launch Jupyter Notebook**
```powershell
jupyter notebook
```

**5️⃣ Run the model**
- Open `Model/brain_tumor_detection_and_classification.ipynb`
- Update `trainPath` and `testPath` variables
- Execute all cells
- Trained model saved as `classification.h5` ✅

---

## 📊 Results & Visualizations

### Model Predictions on Test Set

<div align="center">

| Glioma Tumor | Meningioma Tumor |
|:------------:|:----------------:|
| ![Glioma example](Results/glioma%20tumor.PNG) | ![Meningioma example](Results/meningioma%20tumor.PNG) |

| No Tumor | Pituitary Tumor |
|:--------:|:---------------:|
| ![No tumor example](Results/no%20tumor.PNG) | ![Pituitary example](Results/pituitary%20tumor.PNG) |

</div>

📈 **Training Metrics:** The notebook generates loss and validation loss curves to monitor model performance.

---

## 📂 Project Structure

```
Brain-Tumor-Detection-and-Classification/
│
├── 📄 README.md                          # Project documentation
├── 📁 Dataset/                           # Training & Testing data
│   ├── Training/                         # Training images (4 classes)
│   └── Testing/                          # Test images (4 classes)
├── 📁 Model/                             # Jupyter notebooks
│   └── brain_tumor_detection...ipynb     # Main training notebook
└── 📁 Results/                           # Prediction visualizations
```

---

## 🌐 Deployment Options

### 💻 Local (Windows)
Follow the Quick Start guide above

### ☁️ Cloud Platforms

**Google Colab / Kaggle Notebooks:**
1. Upload this repository or mount it
2. Update `trainPath` and `testPath` to match your environment
3. Run all cells
4. Download the trained model (`classification.h5`)

---

## 💡 Performance Tips

🔧 **Reduce Overfitting:**
- Add stronger data augmentation (rotation, zoom, shift)
- Apply L2 regularization
- Increase dropout rate

🚀 **Boost Accuracy:**
- Use transfer learning (MobileNetV2, ResNet50, VGG16)
- Fine-tune pre-trained models
- Experiment with different architectures
- Increase training data

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

## ⭐ Show Your Support

If this project helped you, please give it a ⭐️!

---

<div align="center">

**Made with ❤️ for Medical AI Research**

*Advancing healthcare through artificial intelligence*

</div>
- Ensure your GPU drivers/CUDA are set up if training with GPU; otherwise, training will run on CPU and take longer.
- Images are resized to 150×150. Higher resolutions and a stronger backbone typically improve accuracy at a compute cost.

## Acknowledgments

This work is inspired by public brain tumor MRI datasets (e.g., “brain-tumor-classification-mri” on Kaggle) and common CNN baselines in Keras.
