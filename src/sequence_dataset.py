import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class SequenceDataset:
    """
    A class used to represent a sequence dataset and preprocess it for training and validation.

    Attributes
    ----------
    sequence : list
        A list of numbers in a sequence.
    split_pct : int, optional
        The percentage of data to be used for training (default is 80).
    X_train : ndarray
        The training input data after preprocessing.
    x_validation : ndarray
        The validation input data after preprocessing.
    y_train : ndarray
        The training output data after preprocessing.
    scaler_x : MinMaxScaler
        The scaler used to normalize the input data.
    scaler_y : MinMaxScaler
        The scaler used to normalize the output data.

    Methods
    -------
    prepare_data():
        Preprocesses the sequence data for training and validation.
    """

    def __init__(self, sequence, split_pct=80):
        """
        Constructs a new SequenceDataset instance.

        Parameters
        ----------
        sequence : list
            A list of numbers in a sequence.
        split_pct : int, optional
            The percentage of data to be used for training (default is 80).
        """

        self.sequence = sequence
        self.split_pct = split_pct
        self.X_train, self.x_validation, self.y_train, self.scaler_x, self.scaler_y = self.prepare_data()

    def prepare_data(self):
        """
        Preprocesses the sequence data for training and validation.

        The method creates a matrix from the input sequence, scales the input and output data using MinMaxScaler,
        and splits the data into training and validation sets according to the given split percentage.

        Returns
        -------
        X_train : ndarray
            The training input data after preprocessing.
        x_validation : ndarray
            The validation input data after preprocessing.
        y_train : ndarray
            The training output data after preprocessing.
        scaler_x : MinMaxScaler
            The scaler used to normalize the input data.
        scaler_y : MinMaxScaler
            The scaler used to normalize the output data.
        """

        # Create record id
        int_id = list(range(len(self.sequence)))

        # Create matrix
        sequence_of_integers = np.column_stack((int_id, self.sequence))
        follow_up_sequence = sequence_of_integers[1:, 1]
        follow_up_sequence = np.array(follow_up_sequence).reshape(-1, 1)

        # Scale all x and y
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        x_all_trans = scaler_x.fit_transform(sequence_of_integers)
        y_trans = scaler_y.fit_transform(follow_up_sequence)

        # Split data
        num_rows = np.shape(x_all_trans)[0]
        split_point = math.trunc(num_rows * (self.split_pct / 100))

        # Create training and validation data sets using split point
        X_train = np.split(x_all_trans, [split_point])[0]
        x_validation = np.split(x_all_trans, [split_point])[1]
        y_train = y_trans[:split_point, :]

        return X_train, x_validation, y_train, scaler_x, scaler_y
