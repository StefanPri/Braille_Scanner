import numpy as np

# Define a function for categorical cross entropy
def categorical_cross_entropy(yTrue, yPred, numSamples):

    # Avoid numerical instability by adding a small epsilon value
    epsilon = 1e-7
    # Clip the predictions between epsilon and 1 - epsilon
    predictions = np.clip(yPred, epsilon, 1 - epsilon)

    # Calculate the loss
    loss = -np.sum(yTrue * np.log(predictions)) / numSamples

    # Return the loss
    return loss

# Define a function for the derivative of categorical cross entropy
def categorical_cross_entropy_prime(yTrue, yPred):
    # Get the number of samples
    numSamples = yTrue.shape[0]
    # Calculate the gradient
    grad = -yTrue / (yPred + 1e-7) / numSamples
    # Return the gradient
    return grad
