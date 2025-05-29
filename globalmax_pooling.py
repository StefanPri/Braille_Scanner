import numpy as np
from layer import baseLayer


class GlobalMax_Pooling(baseLayer):
    def __init__(self):
        pass

    def forward(self, inputData):
        """
        The forward function takes in a single channel image and returns the maximum value of that image.

        :param self: Represent the instance of the class
        :param inputData: Store the input data to the layer
        :return: The maximum value of each channel
        """
        self.inputImage = inputData
        # get dimensions
        self.batchSize, self.numKernelChannels, self.inputHeight, self.inputWidth = inputData.shape
        # set output array
        self.output = np.zeros((self.batchSize, self.numKernelChannels))

        # Reshape the input data to 2D where each row is a channel
        reshapedInput = inputData.reshape(self.batchSize*self.numKernelChannels, -1)
        # Compute the maximum of each row (channel)
        maxValues = np.max(reshapedInput, axis=1)
        # Reshape the max values to match the output shape
        self.output = maxValues.reshape(self.batchSize, self.numKernelChannels)


        return self.output

    def backward(self, outputGradient, learningRate):
        """
        The backward function takes in the gradient of the output and returns
        the gradient of the input. The backward function is called by a layer's
        parent layer, which passes in its own outputGradient. This function then
        computes how much each weight contributed to that error, and multiplies it by
        the learning rate before returning it.

        :param self: Access the class variables
        :param outputGradient: Calculate the inputgradient
        :param learningRate: Update the weights of the layer
        :return: The gradient of the input image
        """
        # Initialize the input gradient with zeros, same shape as the input image
        inputGradient = np.zeros_like(self.inputImage)

        # Loop over each channel
        for c in range(self.numKernelChannels):
            # Find the maximum value in the current channel
            maxVal = np.max(self.inputImage[c])
            # Create a mask where the input image equals the maximum value
            mask = (self.inputImage[c] == maxVal)

            # Multiply the output gradient with the mask and assign it to the input gradient
            inputGradient[c] = outputGradient[c, 0] * mask

        # Return the input gradient
        return inputGradient
