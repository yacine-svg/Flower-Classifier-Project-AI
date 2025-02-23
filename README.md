# Flower Classifier Project

## Overview

This project involves building a deep learning model to classify images of flowers into two categories: **sunflowers** and **tulips**. The model is built using **TensorFlow** and **Keras**, and the dataset consists of 1714 images divided into training and validation sets. The project includes data loading, visualization, model building, training, and evaluation.

---

## Project Structure

The project is organized into the following steps:

1. **Importing Libraries**: Essential libraries such as TensorFlow, Keras, NumPy, Matplotlib, Seaborn, and Pandas are imported.
2. **Loading Dataset**: The dataset is loaded using `keras.utils.image_dataset_from_directory` with a specified batch size and image size.
3. **Visualizing Class Distribution**: The distribution of sunflower and tulip images in the training dataset is visualized using a bar plot.
4. **Building the Model**: A convolutional neural network (CNN) model is built using Keras. The model includes layers for rescaling, convolution, max-pooling, flattening, and dense layers.
5. **Compiling the Model**: The model is compiled using the Adam optimizer, binary cross-entropy loss, and accuracy as the metric.
6. **Training the Model**: The model is trained on the training dataset for 10 epochs, with validation performed on the validation dataset.
7. **Saving the Model**: The trained model is saved to a file named `flower_classifier.h5`.
8. **Prediction and Visualization**: The model is used to predict the class of images from the validation set, and the results are visualized.

---

## Requirements

To run this project, you need the following Python libraries:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Pandas

You can install these libraries using pip:

```bash
pip install tensorflow keras numpy matplotlib seaborn pandas
```

---

## Dataset

The dataset used in this project is located at `C:\FlowersDataset`. It contains **1714 images** divided into two classes: **sunflowers** and **tulips**. The dataset is split into training and validation sets with an 80-20 ratio.

---

## Model Architecture

The model architecture is as follows:

1. **Input Layer**: Accepts images of size 150x150 with 3 color channels.
2. **Rescaling Layer**: Normalizes the pixel values to the range [0, 1].
3. **Convolutional Layers**:
   - First convolutional layer with 32 filters and a 3x3 kernel, followed by a max-pooling layer.
   - Second convolutional layer with 64 filters and a 3x3 kernel, followed by a max-pooling layer.
4. **Flatten Layer**: Flattens the output from the convolutional layers.
5. **Dense Layers**:
   - A dense layer with 128 units and ReLU activation.
   - A dense layer with 1 unit and sigmoid activation for binary classification.

---

## Training

The model is trained for **10 epochs** with the following parameters:

- **Batch Size**: 32
- **Image Size**: 150x150
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Metric**: Accuracy

---

## Results

After training, the model achieves high accuracy on both the training and validation datasets. The training process is visualized using accuracy and loss plots. The model is also used to predict the class of images from the validation set, and the results are displayed alongside the actual labels.

---

## Usage

To use the trained model for classification, load the saved model and pass new images through it:

```python
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("flower_classifier.h5")

# Preprocess the new image (resize, normalize, etc.)
# new_image = ...

# Predict the class
prediction = model.predict(new_image)
if prediction > 0.5:
    print("Predicted: Sunflower")
else:
    print("Predicted: Tulip")
```

---

## Future Work

- **Data Augmentation**: Implement data augmentation techniques to improve model generalization.
- **Hyperparameter Tuning**: Experiment with different hyperparameters to optimize model performance.
- **Multi-Class Classification**: Extend the model to classify more types of flowers.
- **Deployment**: Deploy the model as a web application or API for real-time predictions.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **TensorFlow** and **Keras** for providing the deep learning framework.
- The dataset providers for making the flower images available.

---

This README provides an overview of the project, its structure, and how to use the trained model. For more detailed code and implementation, refer to the provided Python script.
