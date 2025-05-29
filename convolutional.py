import numpy as np
from scipy import signal
from layer import baseLayer


class Convolution( baseLayer ):
    def __init__(self, batchSize,  inputImageShape, filterShape, numKernels, presetWeights, presetBiases, layerNumber, activation):

        """
        The __init__ function initializes the Convolutional Layer object.

        :param self: Represent the instance of the class
        :param inputImageShape: Determine the shape of the input image
        :param filterShape: Determine the size of the filter
        :param numKernels: Specify the number of kernels in the layer
        :param presetWeights: Set the kernels to a specific value
        :param presetBiases: Set the biases of the convolutional layer
        :param activation: Determine what activation function to use
        """
        # get the dimensions of the input image
        inputColourChannels, inputHeight, inputWidth = inputImageShape

        # Set the depth of the kernels to the number of kernels
        self.kernelsDepth = numKernels  
        # Set the shape of the input image
        self.inputShape = inputImageShape  
        # Set the colour depth of the input image
        self.inputColourDepth = inputColourChannels  

        # Set the shape of the output image
        self.outputShape = (batchSize, numKernels, inputHeight - filterShape + 1, inputHeight - filterShape + 1)
        # Set the shape of the kernels
        self.kernelsShape = (batchSize, numKernels, inputColourChannels, filterShape, filterShape)

        # use the preset weight and biases from the trained model
        self.kernels = presetWeights.astype(np.float32)
        self.biases = presetBiases.astype(np.float32)

        # self.kernels = np.random.rand(*self.kernelsShape) -0.5
        # self.biases = np.random.rand(*self.outputShape) -0.5

        # # LeCun Initialization:
        # variance = 1 / numKernels

        # # He Initialization:
        # variance = 2/ numKernels

        # # Xavier/Glorot Initialization:
        # variance = 2 / (inputSize + outputSize)

        # self.kernels = np.random.randn(*self.kernelsShape)*np.sqrt(variance)
        # self.biases = np.zeros(self.outputShape)

        # set the activation function if there is one
        self.activation = activation

        # get the layer Number
        self.layerNumber = layerNumber

    def saveWeights(self, filepath):
        """
        The saveWeights function saves the weights of a convolutional layer to a file.

        :param self: Represent the instance of the class
        :param filepath: Save the weights in a specific location
        :return: The weights and biases of the layer
        """
        # weights file name
        kernelsFilename = f"conv_{self.layerNumber}_kernels.npy"
        # first save the weights in the filepath with the filename
        np.save(filepath + kernelsFilename, self.kernels)

        # biases file name
        biasesFilename = f"conv_{self.layerNumber}_biases.npy"
        # first save the weights in the filepath with the filename
        np.save(filepath + biasesFilename, self.biases)

    def forward(self, inputImage):

        """
        The forward function takes in an input image and outputs the convolved image.
        The function performs the im2col matrix multiplication 
        The output is then passed through a ReLU activation function.

        :param self: Access the attributes of the class
        :param inputImage: Pass the input image to the convolution layer
        :return: The output of the convolution layer with an activation function
        """

        # save the input image that will be used during forward and backward prop
        self.inputImage = inputImage
        # deep copy the biases over to the output image that will get added
        self.outputImage = np.copy(self.biases)
        # call the im2col convolution function with a stride of 1 and no padding 
        self.outputImage = Convolution.im2colConv( self.inputImage, self.kernels, self.biases, 1, 0)
                                                              
        # # Applying ReLU activation if activation is relu
        if self.activation == "relu":
            # if the value is smaller than zero output zero for that value
            self.outputImage = np.maximum(self.outputImage, 0)
        # print("output shape:\n", self.outputImage.shape)
        return self.outputImage

    @staticmethod
    def im2colConv(inputImage, kernels, biases, strideSize, paddingSize):
        # Set the stride size for the convolution
        convStride = strideSize  

        # Get the number of images in the batch
        batchSize = inputImage.shape[0]  
        # Get the number of colour channels in the image
        colourChannels = inputImage.shape[1]  
        # Get the height of the image
        imageHeight = inputImage.shape[2]  
        # Get the width of the image
        imageWidth = inputImage.shape[3]  

        # Reshape the biases for the convolution
        biases = biases[ : , 0 , 0 ]  

        # Get the number of filters in the kernel
        numFilters = kernels.shape[0]  
        # Get the height of the filter
        filterHeight = kernels.shape[1]  
        # Get the width of the filter
        filterWidth = kernels.shape[2]  
 
        # add padding to the input images
        inputImagePadded = np.pad( inputImage, ( (0, 0), (0, 0), (0, 0), (0, 0) ), mode = 'constant' )

        # calculate the new dimensions of input due to padding
        imageHeight += 2 * paddingSize
        imageWidth += 2 * paddingSize

         # calculate the new dimensions of output due to convolution
        outputImageHeight = (imageHeight - filterHeight) // convStride + 1
        outputImageWidth = (imageWidth - filterWidth) // convStride + 1


        # Using clever stride patters the im2col can be performed 
        # create 6D matrix tensor
        matrixShape6D = (colourChannels, filterHeight, filterWidth, batchSize, outputImageHeight, outputImageWidth)
        # 6D strides tensor 
        # This line calculates the strides for the convolution operation. It's a 6D tensor where each dimension corresponds to a specific stride.
        convStridesMatrix = (imageHeight * imageWidth, imageWidth, 1, colourChannels * imageHeight * imageWidth, convStride * imageWidth, convStride)
        
        # Multiply the item size of the input image with the convolution strides matrix to get the strides in bytes
        convStridesMatrix = inputImage.itemsize*np.array(convStridesMatrix)

        # Use numpy's stride tricks to create a view of the input image with the desired strides and shape
        inputImageConvStridesMatrix = np.lib.stride_tricks.as_strided(inputImagePadded, matrixShape6D, convStridesMatrix)

        # Convert the strides matrix to a contiguous array
        inputImageColumns = np.ascontiguousarray(inputImageConvStridesMatrix)

        # Calculate the size of the colour channel
        colourChannelSize = colourChannels * filterHeight * filterWidth
        # Calculate the size of the batch image
        batchsizeImage = batchSize * outputImageHeight * outputImageWidth

        # Reshape the input image columns
        inputImageColumns.shape = (colourChannelSize , batchsizeImage)

        # Apply the singular matrix multiplication calculation and add biases
        # reshape the dimensions of the kernels to allow for dot product same with biases 
        convOutput = kernels.reshape(numFilters, -1).dot(inputImageColumns)
        # add biases
        convOutput += biases.reshape(-1, 1)

        # determine the new output shape that the conv results need sto be reshaped into 
        convOutput.shape = (numFilters, batchSize, outputImageHeight, outputImageWidth)
        # transpose to get the new output dimensions
        finalOutput = convOutput.transpose(1,0,2,3)

        return np.ascontiguousarray(finalOutput)

    def backward(self, outputGradient, learningRate):

        """
        The backward function takes in the gradient of the output and calculates
        the gradient of the input. It also updates weights and biases based on this
        gradient. The backward function is called after each forward pass, so that
        the network can learn from its mistakes.

        :param self: Store the values of the object
        :param outputGradient: Calculate the gradient of the loss function with respect to each kernel
        :param learningRate: Update the weights and biases
        :return: The gradient of the input
        """
        # Initialize the gradient of the kernels with zeros
        filtersGradient = np.zeros(self.kernelsShape)
        # Initialize the gradient of the input with zeros
        inputGradient = np.zeros(self.inputShape)

        # If the activation function is ReLU, set the output gradient to zero where the output image is less than or equal to zero
        if self.activation == "relu":
            outputGradient[self.outputImage <= 0] = 0

        # Loop over the depth of the kernels
        for kernels in range(self.kernelsDepth):
            # Loop over the depth of the input colour
            for channels in range(self.inputColourDepth):
                # Calculate the gradient of the kernels by correlating the input image with the output gradient
                filtersGradient[kernels, channels] = signal.correlate2d( self.inputImage[channels], outputGradient[kernels], "valid" )
                # Calculate the gradient of the input by convolving the output gradient with the kernels
                inputGradient[channels] += signal.convolve2d( outputGradient[kernels], self.kernels[kernels, channels], "full" )

        # Update the biases by subtracting the product of the learning rate and the output gradient
        self.biases = self.biases - learningRate*outputGradient
        # Update the kernels by subtracting the product of the learning rate and the gradient of the kernels
        self.kernels = self.kernels - learningRate*filtersGradient
        # Return the gradient of the input
        return inputGradient
