from sequence_dataset import SequenceDataset
from neural_network import NeuralNetwork
import numpy as np

class ANNRunner:
    """
    A class to train, predict, and save weights for an Artificial Neural Network (ANN) using a given sequence dataset.
    """

    def __init__(self, sequence, epochs=10000, split_pct=80):
        """
        Initializes the ANNRunner with a given sequence and specified training parameters.

        :param sequence: The input sequence data to be used for training and validation.
        :param epochs: The number of epochs for training the ANN. Defaults to 10000.
        :param split_pct: The percentage of the dataset to be used for training. Defaults to 80.
        """

        self.epochs = epochs
        self.dataset = SequenceDataset(sequence, split_pct)
        self.nn = NeuralNetwork()

    def train(self):
        """
        Trains the ANN using the initialized training parameters and dataset.
        """

        for i in range(self.epochs):
            print(f"# {i}\n")
            print("Training Data Input: \n" + str(self.dataset.scaler_x.inverse_transform(self.dataset.X_train)))
            print("Training Data Output: \n" + str(self.dataset.scaler_y.inverse_transform(self.dataset.y_train)))
            print("Training Data Predicted Output: \n" + str(self.dataset.scaler_y.inverse_transform(self.nn.forward(self.dataset.X_train))))
            print("Loss: \n" + str(np.mean(np.square(self.dataset.y_train - self.nn.forward(self.dataset.X_train)))))
            self.nn.train(self.dataset.X_train, self.dataset.y_train)

    def predict(self):
        """
        Performs predictions using the trained ANN on the validation dataset.
        """

        self.nn.predict(self.dataset.x_validation, self.dataset.scaler_x, self.dataset.scaler_y)

    def save_weights(self):
        """
        Saves the weights of the trained ANN.
        """

        self.nn.save_weights()

