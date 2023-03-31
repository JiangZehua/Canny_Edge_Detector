'''
Canny Edge Detection
'''
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from pdb import set_trace as TT


G_0 = np.array(
    [[-1, 0 , 1],
     [-2, 0 , 2],
     [-1, 0 , 1]])

G_1 = np.array(
    [[0, 1, 2],
     [-1, 0, 1],
     [-2, -1, 0]])

G_2 = np.array(
    [[1, 2, 1],
     [0, 0, 0],
     [-1, -2, -1]])

G_3 = np.array(
    [[2, 1, 0],
     [1, 0, -1],
     [0, -1, -2]])

# 7x7 Gaussian Mask
Gaussian_mask = np.array(
    [[1, 1, 2, 2, 2, 1, 1],
     [1, 2, 2, 4, 2, 2, 1],
     [2, 2, 4, 8, 4, 2, 2],
     [2, 4, 8, 16, 8, 4, 2],
     [2, 2, 4, 8, 4, 2, 2],
     [1, 2, 2, 4, 2, 2, 1],
     [1, 1, 2, 2, 2, 1, 1]])


def convolution(img, mask):
    '''
    Convolution Operation
    '''
    # create a zero matrix to store the result
    result = np.zeros((img.shape[0]-mask.shape[0]+1, img.shape[1]-mask.shape[1]+1))
    
    # convolution operation
    for i in range(img.shape[0]-mask.shape[0]+1):
        for j in range(img.shape[1]-mask.shape[1]+1):
            result[i,j] = np.sum(img[i:i+mask.shape[0], j:j+mask.shape[1]] * mask)
    
    # normalize the result using the sum of the mask (140 for Gaussian mask) if necessary
    norm = 1 / (np.sum(np.concatenate(mask))) if np.sum(np.concatenate(mask)) != 0 else 1 
    return norm * result 


def Gradient_Operation(img, g_0, g_1, g_2, g_3):
    '''
    Calculate the gradient magnitude and direction of each pixel using each for filter, and take the maximum value of the response
    '''
    mask_shape = g_0.shape
    # create a zero matrix to store the result
    result = -10000 * np.ones((img.shape[0] - mask_shape[0] + 1, 
                       img.shape[1] - mask_shape[1] + 1),)
    index = - np.ones((img.shape[0] - mask_shape[0] + 1,
                        img.shape[1] - mask_shape[1] + 1))


    for i, g in enumerate([g_0, g_1, g_2, g_3]):
        compare = convolution(img, g) 
        index = np.where(compare > result, i, index) # index of the maximum value of the response
        result = np.maximum(result, compare) # maximum value of the response

    # calculate the gradient direction
    direction = np.zeros((img.shape[0] - mask_shape[0] + 1,
                        img.shape[1] - mask_shape[1] + 1))
    direction[index == 0] = 0
    direction[index == 1] = 45
    direction[index == 2] = 90
    direction[index == 3] = 135

    return result, direction


def Non_Max_Suppression(respond, direction):
    '''
    Non-Maximum Suppression
    '''
    m, n = respond.shape
    result = np.zeros((m-2,n-2))
    
    for i in range(1,m-1):
        for j in range(1,n-1):
            # angle 0
            if direction[i,j] == 0:
                if respond[i,j] >= respond[i,j-1] and respond[i,j] >= respond[i,j+1]:
                    result[i-1,j-1] = respond[i,j]
                else:
                    result[i-1,j-1] = 0

            # angle 45
            elif direction[i,j] == 45:
                if respond[i,j] >= respond[i-1,j+1] and respond[i,j] >= respond[i+1,j-1]:
                    result[i-1,j-1] = respond[i,j]
                else:
                    result[i-1,j-1] = 0

            # angle 90
            elif direction[i,j] == 90:
                if respond[i,j] >= respond[i-1,j] and respond[i,j] >= respond[i+1,j]:
                    result[i-1,j-1] = respond[i,j]
                else:
                    result[i-1,j-1] = 0

            # angle 135
            elif direction[i,j] == 135:
                if respond[i,j] >= respond[i-1,j-1] and respond[i,j] >= respond[i+1,j+1]:
                    result[i-1,j-1] = respond[i,j]
                else:
                    result[i-1,j-1] = 0
    return result


def simple_threshold(img, threshold):
    '''
    Simple Threshold using 25%, 50%, 75% of the maximum value of the response
    '''
    threshold = threshold * np.max(img)
    result = np.zeros(img.shape)
    result[img >= threshold] = 1
    return result
    

def main(img):
    '''
    main function for Canny Edge Detector
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray scale (512, 512)
    img = convolution(img, Gaussian_mask)
    respond, direction = Gradient_Operation(img, G_0, G_1, G_2, G_3)
    suppressed = Non_Max_Suppression(respond, direction)
    for i in [0.25, 0.5, 0.75]:
        result = simple_threshold(suppressed, i)
        # save the result
        cv2.imwrite(f"result_{i}.bmp", result*255)

    


    


    


if __name__ == "__main__":
    image1 ="/Users/zehuajiang/My/CS6643 Computer Vision Spring 2023/Project_1_Canny_Edge_Detector/Barbara.bmp"
    image2 = "/Users/zehuajiang/Desktop/Project_1_Canny_Edge_Detector/Goldhill.bmp"
    image3 = "/Users/zehuajiang/Desktop/Project_1_Canny_Edge_Detector/Peppers.bmp"

    img = cv2.imread(image1) # image size (512, 512, 3)
    main(img)