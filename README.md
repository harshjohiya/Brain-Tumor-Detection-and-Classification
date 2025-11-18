Brain Tumor Detection and Classification (MRI) with CNN

This project trains a Convolutional Neural Network (CNN) to classify brain MRI scans into four categories:

- glioma tumor
- meningioma tumor
- pituitary tumor
- no tumor

It uses Keras/TensorFlow and a simple yet effective CNN trained on the dataset in `Dataset/`.

## Dataset

Folder layout (already included in this repo):

```
Dataset/
	Training/
		glioma_tumor/
		meningioma_tumor/
		no_tumor/
		pituitary_tumor/
	Testing/
		glioma_tumor/
		meningioma_tumor/
		no_tumor/
		pituitary_tumor/
```

Note: The notebook currently references a Kaggle-style path (`../input/brain-tumor-classification-mri/...`). To run locally on this repo, update these two variables in the notebook to:

- `trainPath = "../Dataset/Training"`
- `testPath = "../Dataset/Testing"`

## Model overview

The model is a Sequential CNN with:

- Input size: 150×150×3
- Conv2D blocks with ReLU activations and MaxPool2D
- Dropout for regularization
- Dense(1024, relu) → Dense(4, softmax)
- Optimizer: Adam (lr=0.001)
- Loss: categorical_crossentropy
- Epochs: 40, Batch size: 40
- Simple horizontal flip data augmentation

You can find and run the model in:

- `Model/brain_tumor_detection_and_classification.ipynb`

## Quickstart

1) Create a Python environment and install dependencies (example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install tensorflow numpy pillow scikit-learn matplotlib pandas jupyter
```

2) Launch Jupyter and open the notebook:

```powershell
jupyter notebook
```

3) In the notebook, adjust `trainPath` and `testPath` as noted above, then run all cells. A trained model will be saved as `classification.h5`.

## Results

Example predictions on the test set classes:

| Glioma | Meningioma |
| --- | --- |
| ![Glioma example](Results/glioma%20tumor.PNG) | ![Meningioma example](Results/meningioma%20tumor.PNG) |

| No tumor | Pituitary |
| --- | --- |
| ![No tumor example](Results/no%20tumor.PNG) | ![Pituitary example](Results/pituitary%20tumor.PNG) |

Training also plots the loss/val-loss curve in the notebook.

## Project structure

```
README.md                      # You are here
Dataset/                       # Training/Testing data (4 classes)
Model/brain_tumor...ipynb      # End-to-end training + evaluation notebook
Results/                       # Sample prediction results used in this README
```

## Reproducing locally or in the cloud

- Local (Windows): follow Quickstart above.
- Google Colab/Kaggle: upload this repo (or mount it), ensure `trainPath`/`testPath` point to the correct folders, then run all cells.

## Notes and tips

- If you see overfitting, consider stronger augmentation (rotation/zoom/shift), adding L2 regularization, or using transfer learning (e.g., MobileNetV2, ResNet50) with fine-tuning.
- Ensure your GPU drivers/CUDA are set up if training with GPU; otherwise, training will run on CPU and take longer.
- Images are resized to 150×150. Higher resolutions and a stronger backbone typically improve accuracy at a compute cost.

## Acknowledgments

This work is inspired by public brain tumor MRI datasets (e.g., “brain-tumor-classification-mri” on Kaggle) and common CNN baselines in Keras.
