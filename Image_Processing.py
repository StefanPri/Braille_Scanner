import numpy as np

def RGB_to_Grayscale(frame):
    # get the matrix dimension of the image
    rows = frame.shape[0]
    columns = frame.shape[1]

    # using the average method the image can be changed to grayscale
    # therefore loop through the different channels and find average
    for row in range(rows):
        for column in range(columns):
            frame[row,column] = (1/3)*sum(frame[row,column])

    # remove the last dimension such that 720,1280
    greyFrame = frame[:,:,0]

    return greyFrame

def resize(frame,newHeight,newWidth):
    # input image height and width
    imageWidth, imageHeight = frame.shape[:2]

    # determine rescaling factor for both dimensions
    heightFactor = newHeight / imageWidth
    widthFactor = newWidth / imageHeight

    # create new resized image dimensions
    resizedImage = np.zeros( [newHeight, newWidth, 1] )

    for pixel_i in range(newHeight):
        for pixel_j in range(newWidth):
            resizedImage[pixel_i, pixel_j] = frame[ int(pixel_i/heightFactor), int(pixel_j/widthFactor) ]

    return resizedImage[:,:,0]




