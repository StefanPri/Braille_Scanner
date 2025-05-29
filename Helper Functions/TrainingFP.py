import time

import numpy as np
import os
import cv2
from dense import Dense
from convolution import Convolution
from softmax import Softmax
from leakyReLU import LeakyReLU
from max_pooling import Max_Pooling
from globalmax_pooling import GlobalMax_Pooling
from losses import categorical_cross_entropy, categorical_cross_entropy_prime
from network import training, predict

# load all the weights for the FP cnn
c0 = np.load('conv_1_kernels.npy')
b0 = np.load('conv_1_biases.npy')
c1 = np.load('conv_2_kernels.npy')
b1 = np.load('conv_2_biases.npy')
c2 = np.load('conv_3_kernels.npy')
b2 = np.load('conv_3_biases.npy')

d0 = np.load('dense_1_weight.npy')
db0 = np.load('dense_1_biases.npy')
d1 = np.load('dense_2_weight.npy')
db1 = np.load('dense_2_biases.npy')
d2 = np.load('dense_3_weight.npy')
db2 = np.load('dense_3_biases.npy')

# filepath to where all the classes are situated
filePath = 'Classes/'
numClasses = 63

def loadData(filepath,validationSplit):
    # create folder of where classes are
    classFolderList = os.listdir(filepath)

    # array to hold the images
    x_array = []
    # array to hold the image labels
    y_array = []

    # next we want to loop through each class folder and obtain the images
    for classIndex, iterativeClassesFolder in enumerate(classFolderList):
        # join the filepath and the class folder name
        classFolderDirectory = os.path.join(filePath, iterativeClassesFolder)

        # get the list of files found within the current class folder
        classFiles = os.listdir(classFolderDirectory)

        # next we loop through each file found within the class folder and append
        for imageName in classFiles:
            # first we get the image directory
            imageFilePath = os.path.join(classFolderDirectory, imageName)

            # load the image and change to grayscale
            image = cv2.imread(imageFilePath, cv2.IMREAD_GRAYSCALE)

            # normalise the image
            normalisedImage = image/255

            # scale image to 28*28 pixels
            scaledImage = cv2.resize(normalisedImage,(28,28))

            # append the image to the images array
            x_array.append(scaledImage)

            # append the image label to labels array
            y_array.append(classIndex)

    # now that we have all the images we are able to split the data into testing and validation
    # convert to np arrays for array manipulation
    x_array = np.array(x_array)
    y_array = np.array(y_array)

    # to ensure proper training the data is shuffled
    dataLength = len(x_array)
    indexShuffling = np.random.permutation(dataLength)
    x_array = x_array[indexShuffling]
    x_array = x_array.reshape(len(x_array),28,28,1)
    y_array = y_array[indexShuffling]

    # next we want to be able to split the data into training, testing and validation therefore ratios will be used
    # validation is useful to test the model as the performance can be tested and over fitting can be prevented
    trainingSplit = 0.8
    # adding all the split should add up to one
    validationSplit = validationSplit
    testingSplit = 1.0 - trainingSplit - validationSplit

    # get the index of where training ends and round to nearest number
    trainEndIndex = int( len(x_array) * trainingSplit )
    validationEndIndex = int( len(x_array) *( trainingSplit + validationSplit) )

    # split the data
    x_train = x_array[:trainEndIndex]
    y_train = y_array[:trainEndIndex]

    x_validate = x_array[trainEndIndex:validationEndIndex]
    y_validate = y_array[trainEndIndex:validationEndIndex]

    x_test = x_array[validationEndIndex:]
    y_test = y_array[validationEndIndex:]

    return (x_train,y_train),(x_validate,y_validate),(x_test,y_test)

def extendLabels(labelsArray,classes):
    finalArray = np.zeros((len(labelsArray),numClasses,1))
    counter = 0
    for i in zip(labelsArray):
        # create two dim array
        extendedArray = np.zeros((classes, 1), dtype=float)
        # set the index to one that corresponds to the index
        extendedArray[i, 0] = 1

        # append to the final array
        finalArray[counter, :, :] = extendedArray

        # increment the counter
        counter += 1

    return finalArray
print(time.time())
print("load")
(x_train,y_train),(x_validate,y_validate),(x_test,y_test) = loadData(filePath,0.1)

# extend the labels array to multi dimensional arrays.
print(time.time())
print("labels")
y_train = extendLabels(y_train, numClasses)
print("extend label")
y_validate = extendLabels(y_validate, numClasses)
print("extend label")
y_test = extendLabels(y_test, numClasses)
print("extend label")
print(time.time())

# building the neural network model and calling all constructors
network = [ Convolution( (1, 28, 28), 3, 64, c0, b0, layerNumber=1, activation="relu"),
            Max_Pooling(2),
            Convolution( (64,13,13), 3, 128, c1, b1, layerNumber=2, activation="relu"),
            Max_Pooling(2),
            Convolution( (128, 5, 5), 2, 256, c2, b2, layerNumber=3, activation="relu"),
            GlobalMax_Pooling(),
            Dense(256, 256, d0, db0, layerNumber=1),
            LeakyReLU(),
            Dense(256, 64, d1, db1, layerNumber=2),
            LeakyReLU(),
            Dense(64, 63, d2, db2, layerNumber=3),
            Softmax()
            ]
# fp = 'Training Weights/'
# for layer in network:
#     layer.saveWeights(fp)

# train which should do the forwards and backwards propagation and update the parameters
print("Training start time")
print(time.time())
start_time = time.time()
training(
    network,
    categorical_cross_entropy,
    categorical_cross_entropy_prime,
    x_train,
    y_train,
    x_validate,
    y_validate,
    x_test,
    y_test,
    epochs=5,
    learning_rate=0.01
)
print("traing done")
print(time.time())
end_time = time.time()
print(f"Total Time: {end_time-start_time}")
accuracy = 0
numSamples = len(x_test)

print(f"<=====================Testing of Model=====================>")
print("Testing start time")
print(time.time())
start_time = time.time()

for x, y in zip(x_test, y_test):
    test = predict(network, x)
    if np.argmax(test) == np.argmax(y):
        accuracy += 1
    print(f"predicted: {np.argmax(test)}, true_value: {np.argmax(y)}")

accuracy /= numSamples
print(f"The testing accuracy of the model is: {accuracy*100}")
print(time.time())
end_time = time.time()
print(f"Total Testing Time: {end_time-start_time}")
