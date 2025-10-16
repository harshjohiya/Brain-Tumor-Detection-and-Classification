Brain Tumor Classification using CNN

This project uses a Convolutional Neural Network (CNN) to classify MRI scans of brains into four distinct categories: glioma tumor, meningioma tumor, pituitary tumor, or no tumor. The model is built with Keras and TensorFlow.

📋 Dataset

The dataset is organized into Training and Testing folders, each containing subfolders for the four classes. Each class contains hundreds of MRI images used for training and evaluating the model.

🧠 Model Architecture

The model is a sequential CNN with the following key layers:

Convolutional Layers (Conv2D): To extract features like edges and textures from the images.

Max Pooling Layers (MaxPool2D): To reduce dimensionality and computational load.

Dropout Layers: To prevent overfitting by randomly deactivating neurons during training.

Dense Layers: Fully connected layers for classification, with a final softmax activation layer that outputs the probability for each of the four classes.

📈 Results and Performance

The model was trained for 40 epochs and achieved a high training accuracy of approximately 93%. However, the validation loss plot indicates that the model began to overfit the training data, resulting in a final validation accuracy of around 66%. Despite this, the model can still make highly confident and accurate predictions on certain test images.

🚀 How to Use

Clone the repository:
git clone https://github.com/harshjohiya/Brain-Tumor-Detection-and-Classification.git

Open the brain_tumor_detection_and_classification.ipynb notebook in a Jupyter environment like Kaggle or Google Colab.

Ensure the dataset paths are correct.

Run the cells sequentially to train the model and see predictions on test images.

💡 Future Improvements

To improve the model's performance and reduce overfitting, future work could include:

Implementing more advanced data augmentation techniques.

Using a pre-trained model and applying transfer learning (e.g., VGG16, ResNet).

Fine-tuning hyperparameters like the learning rate and dropout rate.
