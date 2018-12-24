import numpy as np

def class_weighter(labels_encoded):
    '''Takes in a NumPy array contained a set of encoded labels
    Returns a dict in the form {label: weight}'''
    classes_unique = np.array(list(set(labels_encoded)))
    class_weight_dict = {label: 0.75 / len(labels_encoded[labels_encoded == label]) for label in classes_unique}
    return class_weight_dict

def train_test_bootstrapper(data_array, labels_array, bootstrapper_size=None, random_state=0):
    '''
    data_array - NumPy array containing the image data
    labels_array - NumPy array containing the encoded labels of each image
    (no one hot encoding done)
    bootstrapper_size - size of bootstrapped training set to be returned,
    same size as original data_array if set to None
    random_state - random seed number for the RNG
    Returns the training data, test_data, training labels and test labels as NumPy arrays
    '''
    
    if bootstrapper_size == None:
        bootstrapper_size = len(labels_array)
        
    # Get probability array for random sampling
    prob_array = np.array([0.75 if len(labels_array[labels_array == label]) <= 10 else 0.25 for label in labels_array])
    
    #prob_array = np.array([])
    #for label in labels_array:
    #    label_count = len(labels_array[labels_array == label])
    #    if label_count <= 10:
    #        prob_array = np.append(prob_array, 0.75)
    #    else:
    #        prob_array = np.append(prob_array, 0.25)
    
    prob_array = prob_array / np.sum(prob_array)
        
    np.random.seed(random_state)
    index_array = np.arange(len(labels_array))
    train_sample_index = np.random.choice(index_array, bootstrapper_size, replace=True, p=prob_array)
    test_sample_index = index_array[~np.isin(index_array, train_sample_index)]
    data_train, labels_train = data_array[train_sample_index], labels_array[train_sample_index]
    data_test, labels_test = data_array[test_sample_index], labels_array[test_sample_index]
    return data_train, data_test, labels_train, labels_test