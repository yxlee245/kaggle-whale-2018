### Functions to help in preprocessing ###
import numpy as np
import cv2

def image_load_resize(filename, resize_width):
    '''Takes in filename as string and returns a square NumPy array of size resize_width'''
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (resize_width, resize_width), interpolation=cv2.INTER_AREA)
    return image_resized

def class_weighter(labels_encoded):
    '''Takes in a NumPy array contained a set of encoded labels
    Returns a dict in the form {label: weight}'''
    classes_unique = np.array(list(set(labels_encoded)))
    sample_weight_dict = {label: 0.75 / len(labels_encoded[labels_encoded == label]) for label in labels_encoded}
    return sample_weight_dict
    