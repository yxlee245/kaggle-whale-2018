import numpy as np

def class_weighter(labels_encoded):
    '''Takes in a NumPy array contained a set of encoded labels
    Returns a dict in the form {label: weight}'''
    classes_unique = np.array(list(set(labels_encoded)))
    sample_weight_dict = {label: 0.75 / len(labels_encoded[labels_encoded == label]) for label in labels_encoded}
    return sample_weight_dict

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

    # Generate list of probabilities for each label in labels_array
    sample_weight_dict = class_weighter(labels_array)
    prob_list = [sample_weight_dict[label] for label in labels_array]
        
    np.random.seed(random_state)
    index_array = np.arange(len(labels_array))
    train_sample_index = np.random.choice(len(labels_array), bootstrapper_size, replace=True,
    p=prob_list)
    test_sample_index = train_sample_index[~np.isin(index_array, train_sample_index)]
    data_train, labels_train = data_array[train_sample_index], labels_array[train_sample_index]
    data_test, labels_test = data_array[test_sample_index], labels_array[test_sample_index]
    return data_train, data_test, labels_train, labels_test