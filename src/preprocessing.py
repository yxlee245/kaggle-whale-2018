### Functions to help in image preprocessing ###
import numpy as np
import cv2

def image_load_resize(filename, resize_width):
    '''Takes in filename as string and returns a square NumPy array of size resize_width'''
    image = cv2.imread(filename)
    image_resized = cv2.resize(image, (resize_width, resize_width))
    return image_resized