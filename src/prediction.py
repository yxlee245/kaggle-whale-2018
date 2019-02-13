import numpy as np
import pandas as pd

### Function to select top number of elements in an array, in sorted order
def array_filter(array, labels, n_top):
    '''
    array - NumPy array containing softmax probabilities on which sorting is based
    labels - NumPy array with original labels in string format
    n_top - the number of top elements to find
    Returns a NumPy array containing the label strings of the sorted values
    '''
    idx = (-array).argsort()[:n_top]
    labels_new = labels[idx]
    return labels_new

def array_filter_v2(array, label_2_index_dict, n_top, threshold):
    '''
    array - NumPy array containing probabilities on which sorting is based
    label_2_index_dict - Python dict with original labels as keys and corresponding one-hot encoded indices as values
    n_top - the number of top elements to find
    threshold - the probability below which an image would be assumed to be of the 'new_whale' class
    Returns a NumPy array containing the label strings of the sorted values
    '''
    idx = (-array).argsort()[:n_top]
    # Swap order in dict such that the OHE indices are the keys and the original labels are the values
    index_2_label_dict = {value:key for key, value in label_2_index_dict}
    labels_new = np.array([index_2_label_dict[index] for index in idx])
    for i, index, label in zip(range(n_top), idx, labels_new):
        if array[index] < threshold:
            labels_new[i] = 'new_whale'
            break
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

def compute_proximity(array1, array2):
    '''
    Computes proximity (inverse of L2 Norm) between two 1-D arrays array1 and array2
    '''
    return 1 / np.linalg.norm((array1 - array2))

def compute_prob_wrapper(left_array, right_input, model):
    '''
    Get prediction for each class
    '''
    def compute_prob(left_array):
        print('Shape of left_array:', left_array.shape)
        
    return compute_prob
    #left_input = np.tile(left_array, (right_input.shape[0], 1, 1, 1))
    #pred = model.predict([left_input, right_input]).flatten()
    #return pred