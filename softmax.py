import numpy as np
from layer import baseLayer

class Softmax(baseLayer):
    def __init__(self):
        pass

    def forward(self, inputImage):
        """
        The forward function takes in an input and returns the output of a softmax function.
        The softmax function is defined as:
            f(x) = \frac{e^x}{\sum_{i=0}^{n} e^x_i}$
        where n is the number of elements in the input vector. The max trick helps with numerical stability,
        and prevents overflow errors when computing exponentials.

        :param self: Represent the instance of the class
        :param inputImage: Calculate the output
        :return: The softmax of the input
        """

        # Shift the input values to avoid numerical instability
        shifted_inputImage = inputImage - np.max(inputImage, axis=1, keepdims=True)
        # Compute the exponentiated values
        exp_values = np.exp(shifted_inputImage)

        # Compute the sum of exponentiated values using log-sum-exp trick
        sum_exp_values = np.sum(exp_values, axis=1, keepdims=True)

        # Compute the softmax probabilities
        self.output = exp_values / sum_exp_values

        return self.output


    def backward(self, outputGradient, learningRate):
        """
        The backward function computes the gradient of the loss with respect to
        the input. The formula for this is:

        :param self: Refer to the object itself
        :param outputGradient: Calculate the gradient of the loss function with respect to this layer's output
        :param learningRate: Update the parameters of the layer
        :return: The gradient of the loss with respect to the input
        """
        # Get the size of the output
        n = np.size(self.output)
        # Compute the input gradient using the formula
        inputGradient = np.dot( (np.identity(n) - self.output.T) * self.output, outputGradient)
        # Return the computed input gradient
        return inputGradient


