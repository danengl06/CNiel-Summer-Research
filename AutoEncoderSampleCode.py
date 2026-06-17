import numpy as np


# Loads file of data with shape (30,8)
data = np.loadtxt("AutoEncoderData.txt")


# Controls the weight of each update
# A middle ground of a rate that is not too small or large is needed
learning_rate = 0.001

# Number of times the entire data will run
runs = 30



# ENCODER

# Creates an empty weight matrix for the encoder
# This has 8 rows and 4 columns
W1 = np.random.randn(8, 4) * 0.1

# Creates a bias vector full of zeros
# It has 1 row and 4 columns, which is one bias for each hidden neuron
b1 = np.zeros((1,4))

# DECODER

# Creates an empty weight matrix for the decoder
# Has 8 rows and 4 columns
W2 = np.random.randn(4,8) * 0.1

# Creates a bias vector full of zeros for decoding
# Has 1 row and 8 columns, which is one bias for each hidden neuron
b2 = np.zeros((1,8))



# Loops the training for the autoencoder
for run in range(runs):

    # ENCODING PROCESS
    # Main equation used: Z = XW + b

    # Changes data into a matrix of size (30,4)
    # (30, 8) @ (8,4) = (30,4)
    EncodedData = data @ W1 + b1

    # Sets the negative values in the data to zeros
    EncodedData = np.maximum(0,EncodedData)



    # DECODING PROCESS
    # Main equation used: Y = ZW + b

    # Changes data back into matrix of size (30,8)
    # (30, 4) @ (4,8) = (30,8)
    DecodedData = EncodedData @ W2 + b2


    # LOSS CALCULATIONS

    # Calculates the loss with mean squared error (MSE)
    # This will meassure the differcen between the reconstructed
    # data and the origonal data.

    # A smaller value is wanted
    loss = np.mean((data - DecodedData)**2)



    # BACKPROPAGATION


    # Calculating output gradient
    # This tells us how each each output value contributed to the new data
    OutputGrad = 2 * (DecodedData - data) / data.shape[0]


    # DECODER GRADIENTS

    # Multiplies together the two matrixes, gets a Matrix of (4,8)
    DecodeW2 = EncodedData.T @ OutputGrad

    # Adds contribtiions from all 30 samples
    Decodeb2 = np.sum(OutputGrad, axis=0, keepdims=True)


    # BACKPROP HIDDEN LAYER
    # Putes reconsturcted error backward through the decoder

    # Gets matrix size (30,4)
    Backprop = OutputGrad @ W2.T



    # RELU DERIVATIVE
    # 1 if x < 0
    # 0 if x <= 0

    # Inactive hidden neuros will receive no gradient
    Backprop[EncodedData <= 0] = 0



    # ENCODER GRADIENTS
    # Gradients for exact encoder weights
    EncodeW1 = (data - np.mean(data, axis=0)).T @ Backprop
    Encodeb1 = np.sum(Backprop, axis=0, keepdims=True)


    # GRADIENT DESCENT

    # Updates weights in the correct dimension to reduce loss
    # Formula: New Weight = Old Weight - LearningRate x Gradient

    W1 -= learning_rate * EncodeW1
    b1 -= learning_rate * Encodeb1

    W2 -= learning_rate * DecodeW2
    b2 -= learning_rate * Decodeb2

    # Prints runs and the assosiated loss with each

    if run % 2 == 0:
        print(f"run {run:5d}  Loss = {loss:.6f}")




