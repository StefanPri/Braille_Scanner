import numpy as np 
from layer import baseLayer 

class Max_Pooling(baseLayer):
    def __init__(self, poolSize):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the poolSize variable for use in other functions.

        :param self: Represent the instance of the object itself
        :param poolSize: Set the size of the pool
        """
        self.poolSize = poolSize

    def forward(self, inputImage):
        """
        The forward function takes in an input image and returns the output of the pooling layer.
        The function loops through each feature map, then loops through each pixel in that feature map,
        and finally finds the maximum value within a sliding window of size (poolSize x poolSize).
        This is done for every pixel in every feature map.

        :param self: Access the attributes and methods of the class
        :param inputImage: Pass the input image to the forward function
        :return: The output of the pooling layer
        """
        # Storing the input image in the instance variable
        self.inputImage = inputImage
        # Extracting the dimensions of the input image
        self.batchSize, self.numKernelChannels, self.inputHeight, self.inputWidth = inputImage.shape
        # Calculating the dimensions of the output image after pooling
        self.outputHeight = self.inputHeight // self.poolSize
        self.outputWidth = self.inputWidth // self.poolSize

        # Initializing the output array with zeros
        self.output = np.zeros( (self.batchSize, self.numKernelChannels, self.outputHeight, self.outputWidth) )
        # Applying the im2col operation for max pooling using a smart reshape interpretation
        self.output = Max_Pooling.max_pool_im2Col(self.inputImage, poolSize)
        # Returning the output array
        return self.output

    @staticmethod
    def max_pool_im2Col(inputImage, poolSize):
        
        # Extracting the batch size from the input image shape
        batchSize = inputImage.shape[0]
        # Extracting the number of colour channels from the input image shape
        colourChannels = inputImage.shape[1]
        # Extracting the height of the input image from the input image shape
        imageHeight = inputImage.shape[2]
        # Extracting the width of the input image from the input image shape
        imageWidth = inputImage.shape[3]

        # Setting the height of the pooling window
        poolHeight = poolSize
        # Setting the width of the pooling window
        poolWidth = poolSize
        # Setting the stride length for the pooling operation
        strideLength = poolSize

        # Checking if the image height is divisible by the pool height
        if imageHeight % poolHeight != 0:
            # If not, adjusting the image height to be divisible by the pool height
            imageHeight = imageHeight - (imageHeight % poolHeight)
        # Checking if the image width is divisible by the pool width
        if imageWidth % poolWidth != 0:
            # If not, adjusting the image width to be divisible by the pool width
            imageWidth = imageWidth - (imageWidth % poolWidth)

        # Adjusting the input image to the new dimensions
        inputImage = inputImage[:, :, :imageHeight, :imageWidth]

        # Reshaping the input image for the pooling operation
        inputImageReshaped = inputImage.reshape(batchSize, colourChannels, imageHeight//poolHeight, poolHeight, imageWidth//poolWidth, pool_width)
        # Setting the axis for the max operation
        axis1 = 3
        # Setting the axis for the max operation
        axis2 = 4

        # Performing the max operation on the reshaped input image
        outputMatrix = inputImageReshaped.max( axis = axis1 ).max( axis = axis2 )

        # Returning the output matrix
        return outputMatrix

    def backward(self, outputGradient, learningRate):
        """
        The backward function takes in the output gradient from the next layer and computes
        the input gradient for this layer. The input gradient is then used to update weights
        in this layer using stochastic gradient descent.

        :param self: Represent the instance of the class
        :param output_gradient: Determine the gradient of the output layer
        :param learning_rate: Update the weights of the network
        :return: The gradient of the input image
        :doc-author: Trelent
        """

        # Determining the output shape using previous parameters
        inputGradient = np.zeros_like(self.inputImage)

        # Looping through different feature maps
        for colourChannels in range(self.numKernelChannels):
            # Looping through the height of the output
            for i in range(self.outputHeight):
                # looping through the width of the output
                for j in range(self.outputWidth):
                    # Starting position of the sliding window
                    startIndex1 = i * self.poolSize
                    startIndex2 = j * self.poolSize

                    # Ending Position of the sliding window
                    endIndex1 = startIndex1 + self.poolSize
                    endIndex2 = startIndex2 + self.poolSize

                    # Creating the matrix window
                    imagePatch = self.inputImage[colourChannels, startIndex1:endIndex1, startIndex2:endIndex2]

                    mask = imagePatch == np.max(imagePatch)

                    inputGradient[colourChannels, startIndex1:endIndex1, startIndex2:endIndex2] = outputGradient[colourChannels, i, j] * mask

        return inputGradient



