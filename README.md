# NumPy_Project

Here's what you I leart from the video "Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math) by Samson Zhang":

* The basic structure of a neural network, including input, output, and hidden layers.
* The feedforward process, which involves passing input data through the network to produce an output.
* The backpropagation algorithm, which adjusts the weights in the network based on the error between the predicted output and the actual output.
* The importance of activation functions in neural networks, which help introduce non-linearity and make the network more powerful.
* How to implement a neural network using only Python's NumPy library and mathematical equations.
* How to code the feedforward process, backpropagation algorithm, and weight updates in Python using NumPy.
* How to train a neural network to perform a specific task, such as image classification or regression.


## Project Description:
The objective of this project is to build a neural network using only Python's NumPy library and mathematical equations to classify images from the MNIST dataset. The neural network will be trained on a subset of the dataset and then tested on a separate validation set. The final accuracy of the model will be evaluated and reported.

## Project Details:

Data preprocessing: The first step is to preprocess the MNIST dataset by flattening the 28x28 pixel grayscale images into a 784-dimensional feature vector. Additionally, the labels for each image will need to be one-hot encoded.

Neural network architecture: The next step is to define the architecture of the neural network. For this project, a fully connected feedforward network with one hidden layer will be used. The number of nodes in the input layer will be 784 (the number of features in each image), the number of nodes in the hidden layer will be user-defined, and the number of nodes in the output layer will be 10 (one for each digit from 0-9).

Feedforward process: The feedforward process involves passing the input data through the network to produce an output. This will be implemented using mathematical equations and NumPy.

Backpropagation algorithm: The backpropagation algorithm adjusts the weights in the network based on the error between the predicted output and the actual output. This will also be implemented using mathematical equations and NumPy.

Training and validation: The model will be trained on a subset of the MNIST dataset and then validated on a separate validation set. The number of epochs and learning rate will be hyperparameters that can be adjusted to optimize the performance of the model.

Model evaluation: The final accuracy of the model will be evaluated on the validation set and reported.

