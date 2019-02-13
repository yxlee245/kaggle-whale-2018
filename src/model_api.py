### Functions for NN models ###
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation, AveragePooling2D
from keras.optimizers import Adam

def getPretrainedModel(model, num_classes, num_layers_unfreeze):
    '''
    model - A pre-trained model called from keras.applications
    num_classes - Number of output classes
    num_layers_unfreeze - If int, number of pre-trained layers to unfreeze,
    counting from the output layer of the pre-trained model. If float, fraction of pre-trained layers to unfreeze
    Returns a compiled model'''
    
    # Freeze layers
    num_layers = len(model.layers)
    if num_layers_unfreeze == 0:
        for layer in model.layers:
            layer.trainable = False
    elif isinstance(num_layers_unfreeze, int):
        for layer in model.layers[:-num_layers_unfreeze]:
            layer.trainable = False
    else:
        for layer in model.layers[:-int(num_layers*num_layers_unfreeze)]:
            layer.trainable = False
            
    # Add dense and softmax layer
    x = model.output
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Compile model
    model_final = Model(inputs=model.input, outputs=predictions)
    model_final.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model_final

def getPretrainedModelV2(model, num_classes, num_layers_unfreeze, optimizer='adam', fc_size=100, dropout_rate=0.25):
    '''
    model - A pre-trained model called from keras.applications
    num_classes - Number of output classes
    num_layers_unfreeze - If int, number of pre-trained layers to unfreeze,
    counting from the output layer of the pre-trained model. If float, fraction of pre-trained layers to unfreeze
    Returns a compiled model'''
    
    # Freeze layers
    num_layers = len(model.layers)
    if num_layers_unfreeze == 0:
        for layer in model.layers:
            layer.trainable = False
    elif isinstance(num_layers_unfreeze, int):
        for layer in model.layers[:-num_layers_unfreeze]:
            layer.trainable = False
    else:
        for layer in model.layers[:-int(num_layers*num_layers_unfreeze)]:
            layer.trainable = False
            
    # Add dense and softmax layer
    x = model.output
    x = Dense(fc_size, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(fc_size, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Compile model
    model_final = Model(inputs=model.input, outputs=predictions)
    model_final.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model_final

def getPretrainedEmbedModel(model, embedding_size):
    '''
    model - A pretrained model from keras.applications
    embedding_size - Number of embedded features outputted
    Returns a compiled model'''
            
    # Add dense embedding layer
    x = model.output
    predictions = Dense(embedding_size, activation='tanh')(x)
    
    # Compile model
    model_final = Model(inputs=model.input, outputs=predictions)
    model_final.compile(loss='binary_crossentropy', optimizer='adam', metrics=None)
    
    return model_final

def getSimpleModelV1(num_classes, input_shape, metric_list=['accuracy'], optimizer='Adam'):
    '''
    num_classes - Number of output classes
    input_shape - shape of each observation (height x width x n_channels)
    Returns a compiled model
    '''
    model = Sequential()
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=32, kernel_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(strides=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(strides=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metric_list)
    
    return model

def getSimpleModelV2(num_classes, resize_width, n_channel=3, metric_list=['accuracy']):
    '''
    num_classes - Number of output classes
    resize_width - Number of pixels per side
    n_channel - Number of color channels (default to 3 for RGB)
    metric_list - List containing metrics for evaluation
    Returns a compiled model
    '''
    
    model = Sequential()
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(resize_width, resize_width, n_channel)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=metric_list)
    
    return model

def getSimpleModelV3(num_classes, resize_width, n_channel=3, metric_list=['accuracy']):
    '''
    num_classes - Number of output classes
    resize_width - Number of pixels per side
    n_channel - Number of color channels (default to 3 for RGB)
    metric_list - List containing metrics for evaluation
    Returns a compiled model
    '''
    
    model = Sequential()
    
    model.add(Conv2D(32, (7, 7), strides=(1, 1), input_shape=(resize_width, resize_width, n_channel)))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((3, 3)))
    
    model.add(Flatten())
    
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(num_classes, activation='softmax'))
        
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=metric_list)
    
    return model

def getSimpleModelV4(num_classes, resize_width, n_channel=3, metric_list=['accuracy']):
    '''
    num_classes - Number of output classes
    resize_width - Number of pixels per side
    n_channel - Number of color channels (default to 3 for RGB)
    metric_list - List containing metrics for evaluation
    Returns a compiled model
    '''
    
    model = Sequential()
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(resize_width, resize_width, n_channel)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=metric_list)
    
    return model

def getSimpleModelV5(num_classes, resize_width, n_channel=3, metric_list=['accuracy']):
    '''
    num_classes - Number of output classes
    resize_width - Number of pixels per side
    n_channel - Number of color channels (default to 3 for RGB)
    Returns a compiled model
    '''
    model = Sequential()
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     input_shape=(resize_width, resize_width, n_channel)))
    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPool2D(strides=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(strides=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=metric_list)
    
    return model

def getSimpleModelV6(num_classes, resize_width, n_channel=3, metric_list=['accuracy']):
    '''
    num_classes - Number of output classes
    resize_width - Number of pixels per side
    n_channel - Number of color channels (default to 3 for RGB)
    metric_list - List containing metrics for evaluation
    Returns a compiled model
    '''
    
    model = Sequential()
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(resize_width, resize_width, n_channel)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=metric_list)
    
    return model

def getSimpleModelV7(num_classes, resize_width, n_channel=3, metric_list=['accuracy']):
    '''
    num_classes - Number of output classes
    resize_width - Number of pixels per side
    n_channel - Number of color channels (default to 3 for RGB)
    metric_list - List containing metrics for evaluation
    Returns a compiled model
    '''
    
    model = Sequential()
    
    model.add(Conv2D(32, (7, 7), strides=(1, 1), input_shape=(resize_width, resize_width, n_channel)))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((3, 3)))
    
    model.add(Flatten())
    
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(num_classes, activation='softmax'))
        
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=metric_list)
    
    return model

def getSimpleModelV8(num_classes, resize_width, n_channel=3, metric_list=['accuracy']):
    '''
    num_classes - Number of output classes
    resize_width - Number of pixels per side
    n_channel - Number of color channels (default to 3 for RGB)
    metric_list - List containing metrics for evaluation
    Returns a compiled model
    '''
    
    model = Sequential()
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(resize_width, resize_width, n_channel)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=metric_list)
    
    return model


def getCNNLayers(input_layer):
    '''
    num_classes - Number of output classes
    input_shape - shape of each observation (height x width x n_channels)
    Returns a compiled model
    '''
    cnn_layers = Conv2D(filters=32, kernel_size=(7, 7))(input_layer)
    cnn_layers = BatchNormalization()(cnn_layers)
    cnn_layers = Activation('relu')(cnn_layers)
    cnn_layers = Conv2D(filters=32, kernel_size=(5, 5))(cnn_layers)
    cnn_layers = BatchNormalization()(cnn_layers)
    cnn_layers = Activation('relu')(cnn_layers)
    cnn_layers = MaxPool2D(strides=(2, 2))(cnn_layers)
    
    cnn_layers = Conv2D(filters=64, kernel_size=(3, 3))(cnn_layers)
    cnn_layers = BatchNormalization()(cnn_layers)
    cnn_layers = Activation('relu')(cnn_layers)
    cnn_layers = Conv2D(filters=64, kernel_size=(3, 3))(cnn_layers)
    cnn_layers = BatchNormalization()(cnn_layers)
    cnn_layers = Activation('relu')(cnn_layers)
    cnn_layers = MaxPool2D(strides=(2, 2))(cnn_layers)
 
    return cnn_layers