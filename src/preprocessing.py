### Functions to help in preprocessing ###
import numpy as np
import cv2

def image_load_resize(filename, resize_width):
    '''Takes in filename as string and returns a square NumPy array of size resize_width'''
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (resize_width, resize_width), interpolation=cv2.INTER_AREA)
    return image_resized