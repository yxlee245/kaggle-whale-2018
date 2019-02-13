import numpy as np
import pandas as pd

def class_weighter(labels_encoded):
    '''Takes in a NumPy array contained a set of encoded labels
    Returns a dict in the form {label: weight}'''
    classes_unique = np.array(list(set(labels_encoded)))
    class_weight_dict = {label: 0.75 / len(labels_encoded[labels_encoded == label]) for label in classes_unique}
    return class_weight_dict

def sample_weighter(labels_array):
    prob_list = [0.75 if len(labels_array[labels_array == label]) <= 20
                 else 0.50 if (len(labels_array[labels_array == label]) > 20 and len(labels_array[labels_array == label]) <=100)
                 else 0.25 for label in labels_array]
    prob_array = np.array(prob_list)
    prob_array = prob_array / np.sum(prob_array)
    return prob_array

def train_test_bootstrapper(data_array, labels_array, bootstrapper_size=None, class_imbalance=False, random_state=0):
    '''
    data_array - NumPy array containing the image data
    labels_array - NumPy array containing the encoded labels of each image
    (no one hot encoding done)
    bootstrapper_size - size of bootstrapped training set to be returned,
    same size as original data_array if set to None
    class_imbalnce - indication of whether class imbalance is present, default False
    random_state - random seed number for the RNG
    Returns the training data, test_data, training labels and test labels as NumPy arrays
    '''
    
    if bootstrapper_size == None:
        bootstrapper_size = len(labels_array)
        
    index_array = np.arange(len(labels_array))
    
    # Get probability array for random sampling
    if class_imbalance:
        prob_list = [0.75 if len(labels_array[labels_array == label]) <= 10\
                     else 0.50 if (len(labels_array[labels_array == label]) > 10 and len(labels_array[labels_array == label]) <=50)
                     else 0.25 for label in labels_array]
        prob_array = np.array(prob_list)
        prob_array = prob_array / np.sum(prob_array)
        np.random.seed(random_state)
        train_sample_index = np.random.choice(index_array, bootstrapper_size, replace=True, p=prob_array)
        
    else:
        train_sample_index = np.random.choice(index_array, bootstrapper_size, replace=True)
    
    test_sample_index = index_array[~np.isin(index_array, train_sample_index)]
    data_train, labels_train = data_array[train_sample_index], labels_array[train_sample_index]
    data_test, labels_test = data_array[test_sample_index], labels_array[test_sample_index]
    return data_train, data_test, labels_train, labels_test

def train_test_bootstrapper_v2(data_array, labels_array, bootstrapper_size=None, class_imbalance=False, random_state=0):
    '''
    data_array - NumPy array containing the image data
    labels_array - NumPy array containing the encoded labels of each image
    (no one hot encoding done)
    bootstrapper_size - size of bootstrapped training set to be returned,
    same size as original data_array if set to None
    class_imbalnce - indication of whether class imbalance is present, default False
    random_state - random seed number for the RNG
    Returns the training data, test_data, training labels and test labels as NumPy arrays
    '''
    
    if bootstrapper_size == None:
        bootstrapper_size = len(labels_array)
        
    index_array = np.arange(len(labels_array))
    
    # Get probability array for random sampling
    if class_imbalance:
        class_weight_dict = class_weighter(labels_array)
        prob_list = [class_weight_dict[label] for label in labels_array]
        prob_array = np.array(prob_list)
        prob_array = prob_array / np.sum(prob_array)
        np.random.seed(random_state)
        train_sample_index = np.random.choice(index_array, bootstrapper_size, replace=True, p=prob_array)
        
    else:
        train_sample_index = np.random.choice(index_array, bootstrapper_size, replace=True)
    
    test_sample_index = index_array[~np.isin(index_array, train_sample_index)]
    data_train, labels_train = data_array[train_sample_index], labels_array[train_sample_index]
    data_test, labels_test = data_array[test_sample_index], labels_array[test_sample_index]
    return data_train, data_test, labels_train, labels_test

def undersampler(labels_array, validation_size=100, random_state=0):
    '''
    labels_array - NumPy array containing the encoded labels of each image
    validation_size - size of validation set to be sampled
    random_state - random seed number for the RNG
    Returns the indices for training sample and validation set as NumPy arrays
    '''
    index_array = np.arange(len(labels_array))
    classes_unique = np.array(list(set(labels_array)))
    sample_index = np.array([], dtype=int)
    for i, class_label in enumerate(classes_unique):
        indices_array = np.argwhere(labels_array == class_label).flatten()
        np.random.seed(random_state+i)
        sample_index = np.append(sample_index, np.random.choice(indices_array, 1, replace=False))
    out_of_sample_index = index_array[~np.isin(index_array, sample_index)]
    np.random.seed(random_state)
    validation_index = np.random.choice(out_of_sample_index, validation_size, replace=False)
    return sample_index, validation_index

def up_down_sampler(labels_array, validation_size, size_per_class, random_state=0):
    '''
    labels_array - NumPy array containing the encoded labels of each image
    validation_size - size of validation set to be sampled
    size_per_class - number of observations to be sampled per class
    random_state - random seed number for the RNG
    Returns the indices for training sample and validation set as NumPy arrays
    '''
    index_array = np.arange(len(labels_array))
    classes_unique = np.array(list(set(labels_array)))
    sample_index = np.array([], dtype=int)
    for i, class_label in enumerate(classes_unique):
        indices_array = np.argwhere(labels_array == class_label).flatten()
        np.random.seed(random_state+i)
        if (indices_array.shape[0] >= size_per_class):
            sample_index = np.append(sample_index, np.random.choice(indices_array, size_per_class, replace=False))
        else:
            sample_index = np.append(sample_index, np.random.choice(indices_array, size_per_class, replace=True))
    out_of_sample_index = index_array[~np.isin(index_array, sample_index)]
    np.random.seed(random_state)
    validation_index = np.random.choice(out_of_sample_index, validation_size, replace=False)
    return sample_index, validation_index

def resampler(df, target_num, random_state=0):
    '''
    df - Pandas array containing filenames of images and label ID
    target_num - number of observations per class for over/under sampling
    random_state - seed number for random sampling function
    Returns a resampled Pandas dataframe
    '''
    res_df = None
    for grp in df.groupby('Id'):
        # Compute number of observations for current group (label)
        n = grp[1].shape[0]
        # True if n < target_num  (Sampling with replacement), False if n >= target_num (Sampling without replacement)
        rows = grp[1].sample(target_num, replace=(n < target_num), random_state=random_state)
    
        if res_df is None: res_df = rows
        else: res_df = pd.concat((res_df, rows))
    res_df.reset_index(drop=True, inplace=True)
    return res_df