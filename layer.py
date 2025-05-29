
class baseLayer:
    def __init__(self):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines what attributes it has.
        In this case, we have two attributes: input and output. This is the base
        class for all layers

        :param self: Represent the instance of the class
        :return: Nothing
        """
        self.input = None
        self.output = None

    def saveWeights(self,filepath):
        """
        The saveWeights function saves the weights and biases of a layer to a file.
        The function takes in one parameter, which is the path to save the weights and biases.

        :param self: Represent the instance of the class
        :param filepath: Specify the location where the weights and biases are to be saved
        :return: A dictionary of the weights and biases
        """
        pass


    def forward(self, inputImage):
        """
        The forward function takes in a single input and returns the output of the layer.

        :param self: Represent the instance of the class
        :param inputImage: Pass the input data into the network
        :return: The output of the network
        """
        pass

    def backward(self, outputGradient, learningRate):
        """
        The backward function computes the gradient of the loss with respect to
        the input, and updates all parameters.

        :param self: Refer to the object itself
        :param outputGradient: Update the parameters of the layer
        :param learningRate: Update the parameters
        :return: The gradient of the loss with respect to the input
        """
        pass
