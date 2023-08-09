# Handwritten Digit Recognition Neural Network- Dig2AI
This project implements a simple neural network using the NumPy library to recognize handwritten digits from the MNIST dataset. The neural network architecture includes one hidden layer and uses the Rectified Linear Unit (ReLU) activation function for the hidden layer and the SoftMax activation function for the output layer. The network is trained using gradient descent to minimize the cross-entropy loss.

![Screenshot (106)](https://github.com/impragya08/Dig2AI--Neural-Model/assets/84717393/a1736e96-2b28-43d0-8663-460d90d6830a)


# Project Overview
This project aims to demonstrate the implementation of a neural network for handwritten digit recognition. The neural network is built using the NumPy library and includes key components such as data preprocessing, forward and backward propagation, gradient descent, and prediction. The project emphasizes understanding the fundamental concepts of neural networks and implementing them from scratch.

# Dataset
The MNIST dataset is used for training and validation. This dataset consists of 28x28 grayscale images of handwritten digits (0 to 9) and their corresponding labels. The dataset is preprocessed by flattening the images into 784-dimensional feature vectors and normalizing pixel values to the range [0, 1].

# Getting Started
Clone the repository or download the project files.
Ensure you have the required libraries installed, such as NumPy and Matplotlib.
Run the provided Python script to train the neural network.
Neural Network Architecture
The neural network consists of an input layer with 784 nodes (one for each pixel), a hidden layer with ReLU activation, and an output layer with SoftMax activation. The architecture parameters can be adjusted based on the problem requirements.

# Training
The network is trained using gradient descent to minimize the cross-entropy loss. The training process involves forward propagation to compute activations, backward propagation to compute gradients, and updating weights and biases using the calculated gradients and a learning rate.

# Making Predictions
After training, the model can make predictions for new inputs. The provided make_predictions function computes the predicted class labels for a given input by applying the trained neural network.

# Results and Visualization
The project includes visualization of predictions for a few examples using the matplotlib library. Predicted labels are compared with actual labels, and the input images are displayed alongside the predictions.

# Contributing
Contributions to this project are welcome! If you find any issues or want to enhance the code, feel free to submit a pull request. For major changes, please open an issue first to discuss the proposed changes.



