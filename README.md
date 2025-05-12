# Image-Detection
Intel Image Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify natural and urban scene images into six categories using the Intel Image Classification Dataset. The model is trained and evaluated in a Google Colab environment using TensorFlow and Keras.

## Dataset

The dataset was downloaded from [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) and contains images of size 150x150 pixels across six categories:

- Buildings  
- Forest  
- Glacier  
- Mountain  
- Sea  
- Street

The dataset is divided into three subsets:
- `seg_train`: For training
- `seg_test`: For testing
- `seg_pred`: For making predictions

## Model Architecture

The CNN model uses the Keras `Sequential` API and includes:
- Multiple Conv2D and MaxPooling2D layers
- BatchNormalization for training stability
- Dropout layers to prevent overfitting
- Dense layers for classification
- A softmax output layer for multi-class prediction

### Training Enhancements:
- Image augmentation via `ImageDataGenerator`
- EarlyStopping and ReduceLROnPlateau callbacks

## Results
The model achieves strong classification accuracy across all categories. Results include:
- Accuracy and loss plots
- Classification reports
- Sample predictions

*For detailed metrics and visuals, refer to the notebook.*

## Technologies Used
- Python
- Google Colab
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- Kaggle API

## Usage
1. Clone this repository
2. Upload the notebook to Google Colab
3. Provide your Kaggle API key (`kaggle.json`)
4. Run all notebook cells to train and test the model
5. Make sure to download the kaggle dataset!

