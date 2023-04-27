import numpy as np
import os
import time

class NeuralNetwork:
    """
    A class used to represent a simple feedforward neural network.

    Attributes
    ----------
    input_layer_size : int
        The size of the input layer.
    output_layer_size : int
        The size of the output layer.
    hidden_layer_size : int
        The size of the hidden layer.
    W1 : ndarray
        The weights connecting the input layer to the hidden layer.
    W2 : ndarray
        The weights connecting the hidden layer to the output layer.

    Methods
    -------
    forward(X):
        Computes the forward pass of the neural network for the given input X.
    backward(X, y, o):
        Performs backpropagation to update the weights based on the error between output o and target y.
    sigmoid(s):
        Computes the sigmoid function for the given input s.
    sigmoid_prime(s):
        Computes the derivative of the sigmoid function for the given input s.
    train(X, y):
        Trains the neural network using the input X and target y.
    predict(x_validation, scaler_x, scaler_y):
        Predicts the output for the given validation data and prints the results.
    save_weights(file_prefix):
        Saves the weights of the neural network to text files.
    """

    def __init__(self):
        """
        Initializes the neural network with default architecture
        and random weights.
        """

        self.input_layer_size = 2
        self.output_layer_size = 1
        self.hidden_layer_size = 3

        # Weights
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)

    def forward(self, X):
        """
        Computes the forward pass of the neural network for the given input X.

        Parameters
        ----------
        X : ndarray
            The input data.

        Returns
        -------
        o : ndarray
            The output of the neural network.
        """

        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o

    def backward(self, X, y, o):
        """
        Performs backpropagation to update the weights based on the error between output o and target y.

        Parameters
        ----------
        X : ndarray
            The input data.
        y : ndarray
            The target output data.
        o : ndarray
            The output of the neural network.
        """

        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoid_prime(o)

        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoid_prime(self.z2)

        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def sigmoid(self, s):
        """
        Computes the sigmoid function for the given input s.

        Parameters
        ----------
        s : float or ndarray
            The input to the sigmoid function.

        Returns
        -------
        float or ndarray
            The result of applying the sigmoid function to the input.
        """

        return 1 / (1 + np.exp(-s))

    def sigmoid_prime(self, s):
        """
        Computes the derivative of the sigmoid function for the given input s.

        Parameters
        ----------
        s : float or ndarray
            The input to the sigmoid function.

        Returns
        -------
        float or ndarray
            The result of applying the derivative of the sigmoid function to the input.
        """

        return s * (1 - s)

    def train(self, X, y):
        """
        Trains the neural network using the input X and target y.

        Parameters
        ----------
        X : ndarray
            The input data.
        y : ndarray
            The target output data.
        """

        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self, x_validation, scaler_x, scaler_y):
        """
        Predicts the output for the given validation data and prints the results.

        Parameters
        ----------
        x_validation : ndarray
            The validation input data.
        scaler_x : MinMaxScaler
            The scaler used to normalize the input data.
        scaler_y : MinMaxScaler
            The scaler used to normalize the output data.
        """

        print("Predicted data based on trained weights: ")
        print('Validation Data Input: \n' + str(scaler_x.inverse_transform(x_validation)))
        print("Validation Data Predicted Output: \n" + str(scaler_y.inverse_transform(self.forward(x_validation))))


    def save_weights(self):
        # Move back to the project root directory from src
        project_root = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.join(project_root, '..'))

        # Check if the weights directory exists, create it if not
        weights_dir = "weights"
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        # Save weights to the weights directory with a timestamp
        timestamp = int(time.time())
        np.savetxt(os.path.join(weights_dir, f"weights_W1_{timestamp}.csv"), self.W1, delimiter=',')
        np.savetxt(os.path.join(weights_dir, f"weights_W2_{timestamp}.csv"), self.W2, delimiter=',')
