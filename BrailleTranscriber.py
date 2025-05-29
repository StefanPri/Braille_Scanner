# OpenCV library for computer vision tasks such as image processing
import cv2  
# NumPy library for numerical operations on multi-dimensional arrays
import numpy as np  
# Time library for time-related tasks
import time  
# Pathlib for easy handling of paths
from pathlib import Path  
# Google Text-to-Speech library for converting text into speech
from gtts import gTTS  
# Playsound library for playing sounds in Python
from playsound import playsound  

from dense import Dense
from convolutional import Convolution
from softmax import Softmax
from leakyReLU import LeakyReLU
from max_pooling import Max_Pooling
from globalmax_pooling import GlobalMax_Pooling
from network import predict
from spellChecker import spellCheck
from AudioSequencer import audioSequencer
from Image_Processing import resize,RGB_to_Grayscale

# load all the pretrained weights for the FP CNN
# These weights are used within the convolutional layer includes kernel weights and biases
c0 = np.load('conv2d_weight_0.npy')
b0 = np.load('conv2d_weight_1.npy')
c1 = np.load('conv2d_1_weight_0.npy')
b1 = np.load('conv2d_1_weight_1.npy')
c2 = np.load('conv2d_2_weight_0.npy')
b2 = np.load('conv2d_2_weight_1.npy')

# These weights and biases are used within the fully connected layer
d0 = np.load('dense_weight_0.npy')
db0 = np.load('dense_weight_1.npy')
d1 = np.load('dense_1_weight_0.npy')
db1 = np.load('dense_1_weight_1.npy')
d2 = np.load('dense_2_weight_0.npy')
db2 = np.load('dense_2_weight_1.npy')

# this function is used to draw a grid 
def drawGrid(frame,rows, columns, top, bottom, left , right):
    # Calculate the height and width of each cell in the grid
    cellHeight = (bottom-top)/rows
    cellWidth = (right-left)/columns

    # Draw horizontal lines in video feed
    for i in range(rows):
        # increment to next row position
        y = top + i * cellHeight
        # draw horizontal line using cv2 line function
        cv2.line(frame, (left, int(y)), (right, int(y)), color=(0,0,255), thickness=1)

    # Draw vertical lines
    for i in range(columns):
        # increment to the next column
        x = left + i * cellWidth
        # draw vertical line using cv2 line function
        cv2.line(frame, (int(x), top), (int(x), bottom), color=(0,0,255), thickness=1)

    return frame
  
# this function is used to save the images found within the grid also called the RCSA algorithm
def saveGridImages(frame,rows, columns, top, bottom, left , right):
    # Calculate the height and width of each cell in the grid
    cellHeight = (bottom-top)/rows
    cellWidth = (right-left)/columns

    # image counter 
    counter = 0 

    # Save images within the grid
    for i in range(rows):
        for j in range(columns):
            # move in the y direction 
            y1 = top + i * cellHeight
            # move in the x direction 
            x1 = left + j * cellWidth
            # end position of frame 
            y2 = y1 + cellHeight
             # end position of frame 
            x2 = x1 + cellWidth

            # crop the image
            gridImage = frame[int(y1):int(y2), int(x1):int(x2)]

            # save the image 
            cv2.imwrite('Classify/'+str(counter)+'.jpg', gridImage)
            counter +=1

# this function was used to construct words from the classified braille characters
def obtainString(probabilityArray):
    # ground truth array that uses the chars in this array to determine what is classified
    groundTruth = ['!', '#', "'", ',', '-', ';', 'a', 'and', 'ar', 'b', 'c', 'capital', 'ch', 'colon', 'd', 'dot',
                   'dquote', 'e', 'ed', 'en', 'er', 'f', 'for', 'g', 'gg', 'gh', 'h', 'i', 'in', 'ing', 'j', 'k', 'l',
                   'letter', 'm', 'n', 'o', 'of', 'ou', 'oun', 'ow', 'p', 'q', 'question', 'r', 's', 'sh', 'space',
                   'st', 't', 'tcc1', 'tcc2', 'tcc3', 'th', 'the', 'u', 'v', 'w', 'wh', 'with', 'x', 'y', 'z']

    classifiedCharacters = ""

    for i in probabilityArray:
        # obtain index of highest character. 
        maxIndex = np.argmax(i)

        # using ground truth obtain the correct character representation
        # added a check to prevent IndexError
        if maxIndex < len(groundTruth):
            if groundTruth[maxIndex] == "space":
                classifiedCharacters += ' '
            elif groundTruth[maxIndex] == "dot":
                classifiedCharacters += '.'
            elif groundTruth[maxIndex] == "dquote":
                classifiedCharacters += '"'
            elif groundTruth[maxIndex] == "question":
                classifiedCharacters += '?'
            elif groundTruth[maxIndex] == "colon":
                classifiedCharacters += ':'
            else:
                classifiedCharacters += groundTruth[maxIndex]
        else:
            print("Index out of range")

    return classifiedCharacters

# batch size is used within the neural network and is the number of images that propagate through the network
batchSize = 1000

# building the neural network model and calling all constructors
network = [ Convolution( batchSize, (1, 28, 28), 3, 64, c0, b0, layerNumber=1, activation = "relu"),
            Max_Pooling(2),
            Convolution( batchSize, (64,13,13), 3, 128, c1, b1, layerNumber=2,  activation = "relu"),
            Max_Pooling(2),
            Convolution( batchSize, (128, 5, 5), 2, 256, c2, b2, layerNumber=3,  activation = "relu"),
            GlobalMax_Pooling(),
            Dense(batchSize, 256, 256, d0, db0, layerNumber=1),
            LeakyReLU(),
            Dense(batchSize, 256, 64, d1, db1, layerNumber=2),
            LeakyReLU(),
            Dense(batchSize, 64, 63, d2, db2, layerNumber=3),
            Softmax()
            ]

sentence_constructed = ""

# Video Object definition
videoObject = cv2.VideoCapture(1, cv2.CAP_DSHOW)

#Obtain the best possible resolution of the image
maxValue = 10000

# set the video object frame width to the biggest width
videoObject.set(cv2.CAP_PROP_FRAME_WIDTH, maxValue)

# set the video object frame width to the biggest width
videoObject.set(cv2.CAP_PROP_FRAME_HEIGHT, maxValue)

# counter to save images
counter = 0
  
while(True):
      
    # capture a single frame from videoObject
    returned, imageFrame = videoObject.read()

    # copy image frame to use for grid overlay
    selectedFrame = imageFrame.copy()

    # has a key been pressed
    pressedKey = cv2.waitKey(1)

    # declare the top left co-ordinates of the grid image
    left = 367
    top = 24
    topLeft = (left, top)

    # get image dimensions
    imageHeight, imageWidth, colourChannels = imageFrame.shape

    # declare the bottom right co-ordinates of the grid image
    bottom = imageHeight - 58
    right = imageWidth - 311
    bottomRight = (right,bottom)

    # Display the captured frame with a rectangle drawn to show ROI
    cv2.imshow('frame', cv2.rectangle(imageFrame, topLeft, bottomRight, (0, 0, 255), thickness = 1))

    # display the grid overlay on the live video 
    cv2.imshow('frame', drawGrid(imageFrame, 25, 40, top, bottom, left, right))

    # to capture an image hit the c button
    if pressedKey & 0xFF == ord('c'):
        # save the scanned image to a folder called ScannedImage
        cv2.imwrite('/ImageFrame/ScannedImage'+str(counter)+ '.jpg', selectedFrame)
        # call the RCSA algorithm
        saveGridImages(selectedFrame, 25, 40, top, bottom, left, right)
        # increment the character count of images that need to be classified
        counter +=1

    # transcribe the text when T is pressed 
    elif pressedKey & 0xFF == ord('t'):
        # first lets get the images that needs to be classified
        root_dir = Path("Classify/")

        # save all the images in ram to be processed
        paths = sorted(root_dir.iterdir(), key=lambda path: int(path.stem))

        # For each image path, read the image in grayscale, resize it to 28x28, normalize it by dividing by 255.0,
        # expand its dimensions, and convert it to float32 type. Store all these processed images in the 'inputs' list.
        inputs = [np.expand_dims(cv2.resize(cv2.imread(str(item), cv2.IMREAD_GRAYSCALE), (28, 28)) / 255.0,
                                 axis=-1).astype(np.float32) for item in paths]

        # convert to numpy array 
        inputsArr = np.array(inputs)

        # using the predict function that does the forward prop of each layer, classify all the images
        start = time.time()
        prediction1 = predict(network, inputsArr)
        end = time.time()
        print(F"classify time is {end-start} \n")

        # obtain the words from the classified text
        start = time.time()
        sentence_constructed = obtainString(prediction1)
        end = time.time()
        print(F"Constructed text is = {sentence_constructed}")
        print(F"construct time is {end - start} \n")

        # perform the spellCheck on the constructed words and text 
        # Since the spellchecker is trained dont train 
        start = time.time()
        sentence_constructed = " ".join(sentence_constructed.split())
        spellCheck = spellCheck(sentence_constructed,train=False)
        end = time.time()
        print(F"Spellcheck is: {spellCheck}")
        print(F"spelcheck time is {end - start} \n")

        # generate the audio of the spelling correction output
        start = time.time()
        # generate audio this is the faster method compared to the audio sequencer.
        # tts = gTTS(spellCheck)
        # save the audio to a folder 
        # tts.save(
        #     'C:/Users/stefa/Desktop/Universiteit/4TH_YEAR/EPR/A Code/10.Completed CNN VB/MP3 files/first_try.mp3')
        # # play the audio 
        # playsound(
        #     'C:/Users/stefa/Desktop/Universiteit/4TH_YEAR/EPR/A Code/10.Completed CNN VB/MP3 files/first_try.mp3')
        audioSequencer(spellCheck)
        end = time.time()
        print(F"playback time is {end - start} \n")

    # to exit the image capture code hit q
    elif pressedKey & 0xFF == ord('j'):
        break


# After image has been captured release the videObject
videoObject.release()
# The windows need to be destroyed
cv2.destroyAllWindows()