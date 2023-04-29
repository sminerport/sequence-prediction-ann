# Sequence Prediction ANN

This repository contains a Python implementation of a simple Artificial Neural Network (ANN) for predicting the next number in a given sequence.

## Inspiration and Credits

This project is inspired by and modifies the source code available at [Enlight: Neural Network](https://enlight.nyc/projects/neural-network). The original project demonstrates the fundamentals of artificial neural networks and walks through the process of creating a simple ANN.

In our project, we have adapted the code to predict the next number in a given sequence instead. We have also added features such as data normalization, modularized code, and customizable parameters.

## Features

- Modularized code with separate classes for data preparation, neural network architecture, and training
- Uses a feedforward neural network with a single hidden layer
- Employs the Sigmoid activation function
- Trains using backpropagation and Mean Squared Error (MSE) loss function
- Splits input data into training and validation sets
- Normalizes input data using MinMaxScaler from scikit-learn

## Class Diagram

![class diagram](uml/output/class/class.png)

## Sequence Diagram

![sequence diagram](uml/output/sequence/sequence.png)

## Dependencies

- Python 3.x
- NumPy
- scikit-learn

## Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`
4. Input a sequence of numbers separated by spaces when prompted (e.g., 2 4 6 8 10)
5. Observe the ANN's training progress and final predictions based on the validation data

## Customization

You can customize various parameters such as the number of epochs, percentage of data used for training, and the number of neurons in the hidden layer by modifying the constants in the main script.

## License

This project is released under the [MIT License](LICENSE).
