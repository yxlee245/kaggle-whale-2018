import numpy as np
import pandas as pd

### Function to select top number of elements in an array, in sorted order
def array_filter(array, labels, n_top, threshold):
    '''
    array - NumPy array containing softmax probabilities on which sorting is based
    labels - NumPy array containing strings
    n_top - the number of top elements to find
    threshold - softmax probability threshold to decide if whale is new_whale
    Returns a NumPy array containing the label strings of the sorted values
    '''
    idx = (-array).argsort()[:n_top]
    labels_new = labels[idx]
    # Replace last label string with "new_whale" if below threshold
    prob_new = array[idx]
    if prob_new[-1] < threshold:
        labels_new[-1] = 'new_whale'
    return labels_new

def array_to_string(array):
    '''
    array - NumPy array which contains string elements
    Returns a NumPy array containing the concatenated string for each row
    '''
    output_str = ''
    for string in array:
        output_str += string + ' '
    output_str = output_str.strip()
    return output_str