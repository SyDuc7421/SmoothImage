import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import statistics



def AverageSmoothing(image, kernelSize):
    # Shape
    imageRow, imageCol, channels = image.shape

    padHeight = int((kernelSize - 1) / 2)
    padWidth = int((kernelSize - 1) / 2)

    # Padding image
    image = cv2.copyMakeBorder(image, padHeight, padHeight, padWidth, padWidth, cv2.BORDER_DEFAULT)
    
    # Smoothing
    resultImage = np.zeros([imageRow, imageCol, channels], dtype=np.uint8)
    for row in range(imageRow + padWidth):
        for col in range(imageCol + padHeight):
            if (col >= padHeight and row >= padWidth):
                for channel in range(channels):
                    values = []
                    for i in range(kernelSize):
                        for j in range(kernelSize):
                            values.append(image[row - padWidth + i, col - padHeight + j, channel])
                    resultImage[row - padWidth, col - padHeight, channel] = int(np.mean(values))

    return resultImage
    
def GaussianSmoothing(image, kernelSize):

    kernel = GaussianKernel(kernelSize, sigma=math.sqrt(kernelSize))
    return Convolution(image,kernel)

def GaussianKernel(kernelSize, sigma=1):
    kernelSize=(kernelSize)//2
    x, y = np.mgrid[-kernelSize : (kernelSize+1), -kernelSize : (kernelSize+1)]
    normalizationConst = 1 / (2.0 * np.pi * sigma**2)
    kernel2D =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normalizationConst
    return kernel2D

def Convolution(image, kernel):
    #shape
    imageRow, imageCol, channels = image.shape
    kernelRow, kernelCol = kernel.shape

    resultMatrix = np.zeros(image.shape, dtype=np.uint8)

    padHeight = int((kernelRow - 1) / 2)
    padWidth = int((kernelCol - 1) / 2)

    # Padding image
    paddedImage = cv2.copyMakeBorder(image, padHeight, padHeight, padWidth, padWidth, cv2.BORDER_DEFAULT)

    for row in range(imageRow):
        for col in range(imageCol):
            for channel in range(channels):
                resultMatrix[row, col, channel] = np.sum(kernel * paddedImage[row:row + kernelRow, col:col + kernelCol, channel])

    return resultMatrix
    
def MedianSmoothing(image, kernelSize):
    # Shape
    imageRow, imageCol, channels = image.shape

    padHeight = int((kernelSize - 1) / 2)
    padWidth = int((kernelSize - 1) / 2)

    # Padding image
    image = cv2.copyMakeBorder(image, padHeight, padHeight, padWidth, padWidth, cv2.BORDER_DEFAULT)
    
    # Smoothing
    resultImage = np.zeros([imageRow, imageCol, channels], dtype=np.uint8)
    for row in range(imageRow + padWidth):
        for col in range(imageCol + padHeight):
            if (col >= padHeight and row >= padWidth):
                for channel in range(channels):
                    values = []
                    for i in range(kernelSize):
                        for j in range(kernelSize):
                            values.append(image[row - padWidth + i, col - padHeight + j, channel])
                    resultImage[row - padWidth, col - padHeight, channel] = int(statistics.median(values))

    return resultImage




if __name__ == "__main__":
    #load image
    image = cv2.imread('Lenna.jpg')
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #smoothing
    # imageResultMean = GaussianSmoothing(imageRGB,3)
    imageResultMean = MedianSmoothing(imageRGB,5)
    # imageResultMean = AverageSmoothing(imageRGB,5)
    #compare result
    print(imageResultMean)
    print(imageResultMean.shape)
    fig = plt.figure()
    fig.set_figheight(4)
    fig.set_figwidth(12)
    fig.add_subplot(1,2,1)

    plt.xticks([]), plt.yticks([])
    plt.title("Image source")
    plt.imshow(imageRGB, cmap='gray')
    fig.add_subplot(1,2,2)
    plt.title("Image result")
    plt.imshow(imageResultMean, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show(block=True)