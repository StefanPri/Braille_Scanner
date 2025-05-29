import numpy as np
from layer import baseLayer

class Dense(baseLayer):
    def __init__(self, batchSize, inputSize, outputSize, presetWeights, presetBiases, layerNumber, l2_lambda = 2e-4):

        """
        The __init__ function initializes the weights and biases of the neural network.
        The inputSize is the number of inputs to this layer, outputSize is the number of outputs from this layer,
        presetWeights are a set of weights that can be used to initialize this layer (if None then random initialization),
        and presetBiases are a set of biases that can be used to initialize this layer (if None then random initialization).

        :param self: Represent the instance of the class
        :param inputSize: Determine the size of the input layer
        :param outputSize: Set the number of neurons in the output layer
        :param presetWeights: Set the weights of the network to a specific value
        :param presetBiases: Set the biases of the network
        :param l2_lambda: Set the lambda value for l2 regularization
        """
        self.weights = presetWeights.astype(np.float32)
        self.biases = presetBiases.astype(np.float32)

        # self.weights = np.random.rand(outputSize, inputSize) - 0.5
        # self.biases = np.random.rand(outputSize, 1) - 0.5

        # # LeCun Initialization:
        # variance = 1 / inputSize

        # # He Initialization:
        # variance = 2/ inputSize

        # # Xavier/Glorot Initialization:
        # variance = 2 / (inputSize + outputSize)

        # self.weights = np.random.randn(outputSize, inputSize)*np.sqrt(variance)
        # self.biases = np.zeros((outputSize,1))

        self.l2_lambda = l2_lambda

        # save the layer number
        self.layerNumber = layerNumber

    def saveWeights(self,filepath):
        """
        The saveWeights function saves the weights and biases of a dense layer to a filepath.

        :param self: Represent the instance of the class
        :param filepath: Specify the directory where the weights and biases will be saved
        :return: The weights and biases of the layer
        """

        # weights file name
        weightsFilename = f"dense_{self.layerNumber}_weight.npy"
        # first save the weights in the filepath with the filename
        np.save(filepath + weightsFilename, self.weights)

        # biases file name
        biasesFilename = f"dense_{self.layerNumber}_biases.npy"
        # first save the weights in the filepath with the filename
        np.save(filepath + biasesFilename, self.biases)

    def forward(self, inputImage):
        """
        The forward function takes in an input image and returns the output of a linear layer.

        :param self: Represent the instance of the class
        :param inputImage: Store the input image
        :return: The dot product of the weights and input image, plus the biases
        """
        # Store the input image
        self.inputImage = inputImage
        # Return the dot product of the input image and the transpose of the weights, plus the biases
        return np.dot( self.inputImage, self.weights.T ) + self.biases[0]

    def backward(self, outputGradient, learningRate):

        """
        The backward function computes the gradient of the loss with respect to
        the weights and biases. It then updates them using a learning rate.

        :param self: Represent the instance of the class
        :param outputGradient: Calculate the gradient of the weights
        :param learningRate: Update the weights and biases
        :return: The input gradient
        """
        # Compute the input gradient by taking the dot product of the transpose of the weights and the output gradient
        inputGradient = np.dot(self.weights.T, outputGradient)

        # Compute the weights gradient by taking the dot product of the output gradient and the transpose of the input image
        weightsGradient = np.dot(outputGradient, self.inputImage.T)
        
        # Add L2 regularization term to the weights gradient
        weightsGradient += 2*self.l2_lambda*self.weights

        # Update the biases by subtracting the product of the learning rate and the output gradient
        self.biases = self.biases - learningRate*outputGradient

        # Update the weights by subtracting the product of the learning rate and the weights gradient
        self.weights = self.weights - learningRate*weightsGradient
        
        return inputGradient
