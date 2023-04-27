import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler

EPOCHS = 10000
PCT_TRAINING = 80


class neural_network(object):
    def __init__(self):
        # parameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # weights
        # weight matrix of dimension (size of layer l, size of layer l-1)

        # weight matrix from input to hidden layer
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        # weight matrix from hidden to output layer
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

        # bias vector of dimension (size of layer l, 1)
        # bias vector from input to hidden layer
        # self.b1 = np.zeros((1, self.hiddenLayerSize))
        # bias vector from hidden to output layer
        # self.b2 = np.zeros((self.inputLayerSize, 1))

    def forward(self, X):
        # forward propagation through our network
        # dot product of X (input) and first set of weights and bias
        self.z = np.dot(X, self.W1)
        # activation function
        self.z2 = self.sigmoid(self.z)
        # dot product of hidden layer (z2) and second set of weights and bias
        self.z3 = np.dot(self.z2, self.W2)
        # final activation function
        o = self.sigmoid(self.z3)
        return o

    def backward(self, X, y, o):
        # backward propagate through the network
        # error in output
        self.o_error = y - o
        # applying derivative of sigmoid to error
        self.o_delta = self.o_error*self.sigmoidPrime(o)

        # z2 error: how much our hidden layer weights contributed to output error
        self.z2_error = self.o_delta.dot(self.W2.T)
        # applying derivative of sigmoid to z2 error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)

        # adjusting first set (input --> hidden) weights
        self.W1 += X.T.dot(self.z2_delta)
        # adjusting second set (hidden --> output) weights
        self.W2 += self.z2.T.dot(self.o_delta)

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self):
        print("Predicted data based on trained weights: ")
        print('Validation Data Input: \n' +
              str(scaler_x.inverse_transform(x_validation)))
        # print("Input (scaled): \n" + str(x_validation))
        print("Validation Data Predicted Output: \n" +
              str(scaler_y.inverse_transform(self.forward(x_validation))))

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")


# sequence is the user input
print()
print()
print("**************************************************************************************")
print("*                                                                                    *")
print("*        ARTIFICIAL NEURAL NETWORK (ANN) TO GUESS THE NEXT NUMBER IN A SERIES        *")
print("*                                                                                    *")
print("**************************************************************************************")
print("*                                                                                    *")
print("* Source:                                                                            *")
print("*    The program modifies the source code available below:                           *")
print("*                https://enlight.nyc/projects/neural-network                         *")
print("*                                                                                    *")
print("**************************************************************************************")
print("* User Input (Below):                                                                *")
print("*    At the prompt below please input a sequence of numbers, each separated by a     *")
print("*    space before pressing 'Enter' (e.g., 6 8 10 12 14 16 18).                       *")
print("*                                                                                    *")
print("* Default Values:                                                                    *")
print("*    EPOCHS     = 1,000                                                              *")
print("*    Training   = 80%                                                                *")
print("*    Validation = 20%                                                                *")
print("*                                                                                    *")
print("* Activation Function:                                                               *")
print("*    Sigmoid                                                                         *")
print("*                                                                                    *")
print("* Loss Function:                                                                     *")
print("*    Mean Square Error (MSE)                                                         *")
print("*                                                                                    *")
print("* ANN Layers:                                                                        *")
print("*    * Converted Input: 2 Neurons (X, N)                                             *")
print("*    *          Hidden: 3 Neurons                                                    *")
print("*    *          Output: 1 Neuron (prediction)                                        *")
print("*                                                                                    *")
print("**************************************************************************************")
print("*                                                                                    *")
print("* Instructions & Methodology:                                                        *")
print("*     Input a series of numbers each separated by a space (e.g., 2 4 6 8 10).        *")
print("*     When finished inputing numbers press 'Enter'. The Artificial Neural            *")
print("*     Network (ANN) converts the input to a series of matrix rows and                *")
print("*     calculates a predicted outcome for each (x, y) coordinate pair                 *")
print("*     (e.g., (0,2),(1,4),(2,6),(3,8),(4,10) ) over 1,000 epochs.  Users can update   *")
print("*     the EPOCHS and PCT_TRAINING const variables set to 1000 and 80 by default,     *")
print("*     respectively.  The PCT_TRAINING variable splits the input data into 80%        *")
print("*     training and 20% validation datasets.  The ANN applies the sigmoid activation  *")
print("*     function for every hidden layer during forward-propagation and applies the     *")
print("*     derivative of this function (e.g., sigmoid prime) for every hidden layer       *")
print("*     during back-propagation.  The ANN contains three layers: (a) input layer,      *")
print("*     (b) hidden layer, and (c) output layer.  The input layer contains 2 neurons:   *")
print("*     one for each x and y coordinate.  The hidden layer contains a default of 3     *")
print("*     neurons but can be updated to any number.  The output layer contains 1 neuron, *")
print("*     which represents the ANN's prediction for a given set of inputs (e.g., X and   *")
print("*     Y coordinates).  The ANN calculates the loss and updates the W1 and W2         *")
print("*     matrices based on the output from this loss function.  Once the ANN has        *")
print("*     completed the specified number of training epochs, it uses the validation data *")
print("*     to make predictions based on the newly updated training weights.               *")
print("*                                                                                    *")
print("* Output:                                                                            *")
print("*     The ANN outputs the training data, its actual output (based on the input),     *")
print("*     the ANN's prediction, and the Loss for each epoch of 1,000.  The Mean Square   *")
print("*     Error (MSE) is used to calculate the loss, which users can see decrease over   *")
print("*     successive iterations as the ANN applies the loss function to adjust the       *")
print("*     weight matrices via matrix multiplication, accordingly. Once the ANN has       *")
print("*     completed the specified number of training epochs (default is 1,000) the ANN   *")
print("*     outputs predictions based on the validation data using the newly trained       *")
print("*     weights.                                                                       *")
print("*                                                                                    *")
print("**************************************************************************************")
print()
seq = [int(x) for x in input(
    'Input a series of numbers separated by spaces (Press enter when done): ').split()]

# create record id
int_id = list(range(len(seq)))

# create matrix
sequence_of_integers = np.column_stack((int_id, seq))
# slice matrix on second value
follow_up_sequence = sequence_of_integers[1:, 1]
follow_up_sequence = np.array(follow_up_sequence)
follow_up_sequence = follow_up_sequence.reshape(
    follow_up_sequence.shape[0], -1)

# all x and y
x_all_orig = np.array((sequence_of_integers), dtype=float)
y_orig = np.array((follow_up_sequence), dtype=float)  # output

# scale all x and y
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_all_trans = scaler_x.fit_transform(x_all_orig)
y_trans = scaler_y.fit_transform(y_orig)

# split data
num_rows = np.shape(x_all_trans)[0]
splitPoint = math.trunc(num_rows * (PCT_TRAINING / 100))

# create training and validation data sets using split point
X_train = np.split(x_all_trans, [splitPoint])[0]
x_validation = np.split(x_all_trans, [splitPoint])[1]
y_to_pass_to_train_function = y_trans[:splitPoint, :]

nn = neural_network()

for i in range(EPOCHS):  # trains the nn
    print("# " + str(i) + "\n")
    print("Training Data Input: \n" + str(scaler_x.inverse_transform(X_train)))
    print("Training Data Output: \n" +
          str(scaler_y.inverse_transform(y_to_pass_to_train_function)))
    print("Training Data Predicted Output: \n" +
          str(scaler_y.inverse_transform(nn.forward(X_train))))

    # mean squared error
    print("Loss: \n" + str(np.mean(np.square(y_to_pass_to_train_function - nn.forward(X_train)))))
    # print("\n")
    nn.train(X_train, y_to_pass_to_train_function)

nn.saveWeights()
nn.predict()
