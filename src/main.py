from ann_runner import ANNRunner


def display_instructions():
    print()
    print("**************************************************************************************")
    print("*                                                                                    *")
    print("*   ARTIFICIAL NEURAL NETWORK (ANN) FOR PREDICTING THE NEXT NUMBER IN A SEQUENCE     *")
    print("*                                                                                    *")
    print("**************************************************************************************")
    print("*                                                                                    *")
    print("* Source:                                                                            *")
    print("*    The program is inspired by and modifies the source code available at:           *")
    print("*                https://enlight.nyc/projects/neural-network                         *")
    print("*                                                                                    *")
    print("**************************************************************************************")
    print("* User Input:                                                                        *")
    print("*    Enter a sequence of numbers separated by spaces (e.g., 6 8 10 12 14 16 18)      *")
    print("*    and press 'Enter'. The ANN will attempt to predict the next number in the       *")
    print("*    sequence based on the input.                                                    *")
    print("*                                                                                    *")
    print("* Default Parameters:                                                                *")
    print("*    Training Epochs     = 10,000                                                    *")
    print("*    Training Data       = 80%                                                       *")
    print("*    Validation Data     = 20%                                                       *")
    print("*                                                                                    *")
    print("* ANN Structure:                                                                     *")
    print("*    1. Input Layer:   2 Neurons (X, N)                                              *")
    print("*    2. Hidden Layer:  3 Neurons (Configurable)                                      *")
    print("*    3. Output Layer:  1 Neuron  (Prediction)                                        *")
    print("*                                                                                    *")
    print("* Activation Function: Sigmoid                                                       *")
    print("* Loss Function:       Mean Square Error (MSE)                                       *")
    print("*                                                                                    *")
    print("**************************************************************************************")
    print("* Overview:                                                                          *")
    print("*    This program uses an Artificial Neural Network (ANN) to predict the next        *")
    print("*    number in a sequence based on the user's input. The ANN consists of three       *")
    print("*    layers: an input layer, a hidden layer, and an output layer. The input layer    *")
    print("*    has 2 neurons, the hidden layer has a default of 3 neurons (configurable), and  *")
    print("*    the output layer has 1 neuron that represents the prediction.                   *")
    print("*                                                                                    *")
    print("*    During the training process, the ANN performs forward and backward              *")
    print("*    propagation to adjust its weights based on the input data. It uses the          *")
    print("*    Sigmoid activation function for hidden layers during forward-propagation and    *")
    print("*    its derivative during backward-propagation.                                     *")
    print("*                                                                                    *")
    print("*    The Mean Square Error (MSE) loss function is used to evaluate the performance   *")
    print("*    of the ANN during training. After a specified number of training epochs         *")
    print("*    (iterations), the ANN uses the trained weights to make predictions based on     *")
    print("*    validation data.                                                                *")
    print("*                                                                                    *")
    print("**************************************************************************************")
    print()


def main():
    display_instructions()
    seq = [int(x) for x in input(
        'Input a series of numbers separated by spaces (Press enter when done): ').split()]

    # Initialize the ANNRunner with the sequence data and training parameters
    runner = ANNRunner(sequence=seq, epochs=10000, split_pct=80)

    # Train the ANN
    runner.train()

    # Make predictions using the trained ANN
    runner.predict()

    # Save the weights of the trained ANN
    runner.save_weights()


if __name__ == "__main__":
    main()
