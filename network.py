import numpy as np
import time

def training(network, lossFunction, lossFunctionPrime, x_label, y_label, x_validate, y_validate, epochs , learning_rate, verbose = True):
    counter = 0
    # Loop over the epochs
    for epoch in range(epochs):  
        # Initialize error to 0 for each epoch
        error = 0  
        # Start time for epoch
        start = time.time()  
        # Loop over each training sample
        for x, y in zip(x_label, y_label):  
            # Increment counter for each sample
            counter +=1  

            # Predict the output for current sample
            output = predict(network, x)  

            # Get the total number of samples
            numSamples = len(x_label)
            # Calculate the loss value
            lossValue = lossFunction(y, output,numSamples)  
            # Accumulate the error
            error += lossValue  
            # Print the loss value
            print(lossValue)  

            # Calculate the gradient of the loss function
            grad = lossFunctionPrime(y, output)  

            # Loop over each layer in the network in reverse order
            for layer in reversed(network):  
                # Backpropagate the gradient through the layer
                grad = layer.backward(grad, learning_rate)  

            # Print the counter
            print(counter)  

        # End time for epoch
        end = time.time()  

        # If verbose is True, print the details
        if verbose:  
            # Print the epoch details
            print(f"<=====================Error for Epoch {epoch}=====================>")  
            # Print the error and time for the epoch
            print(f"{epoch + 1}/{epochs}, error={error}, time = {end-start}")  
            # Print a newline
            print("/n")  

        # Initialize accuracy to 0 for each epoch
        accuracy = 0  
        # Get the total number of validation samples
        numSamples = len(x_validate)  

        # Print the validation details
        print(f"<=====================Validation Testing for Epoch {epoch}=====================>")  
        # Initialize newError to 0 for each epoch
        newError = 0  
        # Loop over each validation sample
        for x, y in zip(x_validate, y_validate):  
            # Predict the output for current validation sample
            test = predict(network, x)

            # Calculate the validation loss
            lossValidate = lossFunction(y,test,numSamples)
            # Accumulate the validation error
            newError += lossValidate

            # If the prediction is correct, increment the accuracy
            if np.argmax(test) == np.argmax(y):
                accuracy += 1
            # Print the predicted and true values
            print(f"predicted: {np.argmax(test)}, true_value: {np.argmax(y)}")

        print(f"<=====================Validate Error for Epoch {epoch}=====================>")
        print(f"{epoch + 1}/{epochs}, error={newError}")

        print(f"<=====================Validation Accuracy Epoch {epoch}=====================>")
        accuracy /= numSamples
        print(f"The accuracy of the model for epoch {epoch} is: {accuracy * 100}")

    # next we need to save all the weights of the model
    filepath = "Training Weights/"
    for layer in network:
        layer.saveWeights(filepath)


def predict( network, input ):
    # Transpose the input array to match needed input dimensions
    tp = np.transpose(input, (0, 3, 2, 1))
    # Initialize the output as the transposed input
    output =tp
    # Loop through each layer in the network
    for layer in network:
        # Forward propagate the output through the current layer
        output = layer.forward(output)
    # Return the final output after passing through all layers
    return output
