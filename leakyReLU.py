import numpy as np
from layer import baseLayer

class LeakyReLU(baseLayer):
    def __init__(self, alpha=0.3):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the parameters of the layer, and initializes them to some default values.
        In this case, we set alpha to 0.3 by default.

        :param self: Represent the instance of the class
        :param alpha: Control the learning rate of the model
        """
        self.alpha = alpha
        self.inputImage = None

    def forward(self, inputImage):
        """
        The forward function takes in an input and returns the output of a ReLU
        activation function. The ReLU activation function is defined as:

        :param self: Represent the instance of the class
        :param input: Pass the input data to the layer
        :return: The input if the input is greater than or equal to 0, otherwise it returns alpha * input
        """
        # Store the input image for backward propagation
        self.inputImage = inputImage
        # Apply the leaky ReLU activation function
        return np.where(inputImage >= 0, inputImage, self.alpha * inputImage)

    def backward(self, outputGradient, learningRate):
        """
        The backward function computes the gradient of the loss with respect to its input,
        given a gradient flowing in from the upper layer.
        Since this is (for now) an elementwise multiplication operation,
        the gradient of the loss with respect to this layer’s input is simply output_gradient * mask.
        The mask is defined as np.where(self.input &gt;= 0, 1, self.alpha), which equals 1 for all positive inputs and alpha otherwise.

        :param self: Store the input and output of the layer
        :param outputGradient: Calculate the gradient of the loss function with respect to this layer's output
        :param learningRate: Update the weights of the layer
        :return: The gradient of the input
        """
        # Compute the gradient of the loss with respect to the input
        inputGradient = outputGradient * np.where(self.inputImage >= 0, 1, self.alpha)
        return inputGradient
