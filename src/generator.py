import numpy as np

import cv2, skimage
from keras.utils import Sequence, np_utils
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input


class DataGenerator(Sequence):
    '''Generates data for the predict_generator method of the Keras Sequential or Model class'''
    def __init__(self, dataframe, x_col, y_col, batch_size, target_size, directory, classes_array=None, rotation_range=0, horizontal_flip=False,
                 preprocess_for='resnet50', random_state=0):
        '''
        dataframe - Pandas dataframe containing filenames (Image) and labels (Id)
        x_col - string of column name in dataframe for filenames
        y_col - string of column name in dataframe for labels
        target_size - tuple of shape of image to be resized to
        directory - string of path to folder containing the images (don't end with slash)
        classes_list - NumPy array of strings for each class label
        preprocess_function - function to be performed on image after resizing
        random_state - seed number for random data augmentation
        '''
        image_filenames = list(dataframe[x_col])
        self.x = [directory + '/' + file_name for file_name in image_filenames]
        self.y_col = y_col
        self.batch_size = batch_size
        self.target_size = target_size
        self.classes_array = classes_array
        self.rotation_range = rotation_range
        self.horizontal_flip = horizontal_flip
        self.preprocess_for = preprocess_for
        self.random_state = random_state
        print('Total no of observations in sequence:', len(self.x))
        if self.y_col != None:
            try:
                if self.classes_array == None:
                    self.classes_array = np.array(list(set(dataframe[self.y_col])))
            except:
                pass
            classes_encoded_array = np.arange(len(self.classes_array))
            y_original = np.array(dataframe[self.y_col])
            y_encoded = np.concatenate([classes_encoded_array[np.argwhere(classes_array == label).flatten()]\
                                        for label in y_original])
            self.y = y_encoded
            self.num_classes = len(self.classes_array)
        self.n = len(self.x)
        
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.y_col != None:
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images_list = []
        for i, file_name in enumerate(batch_x):
            image = skimage.io.imread(file_name)
            image = skimage.color.gray2rgb(image)
            # Perform random data augmentation
            np.random.seed(self.random_state + idx + i)
            rotation_angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            image = skimage.transform.rotate(image, rotation_angle)
            # Randomize the occurence of horizontal flip
            if self.horizontal_flip and np.random.random_integers(0, 1):
                    image = image[:, ::-1]
            image = skimage.transform.resize(image, self.target_size, anti_aliasing=True, mode='constant')
            images_list.append(image)
        
        images_array = np.array(images_list)
        if self.preprocess_for == 'resnet50':
            images_array = resnet50_preprocess_input(images_array)
        
        if self.y_col != None:
            y_onehot = np_utils.to_categorical(batch_y, num_classes=self.num_classes)
            return (images_array, y_onehot)
        else:
            return images_array