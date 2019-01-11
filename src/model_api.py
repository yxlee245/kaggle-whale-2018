### Functions for NN models ###
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.optimizers import Adam

def getPretrainedModel(model, num_classes, num_layers_unfreeze):
    '''
    model - One of the following: ResNet50, DenseNet201, MobileNetV2 or NASNetMobile
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

def getSimpleModel(num_classes, resize_width, n_channel=3, metric_list=['accuracy']):
    '''
    num_classes - Number of output classes
    resize_width - Number of pixels per side
    n_channel - Number of color channels (default to 3 for RGB)
    Returns a compiled model
    '''
    model = Sequential()
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(resize_width, resize_width, n_channel)))
    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPool2D(strides=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(strides=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=metric_list)
    
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
    
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu', input_shape=(resize_width, resize_width, n_channel)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3)))
    
    model.add(Conv2D(filters=8, kernel_size=(5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=8, kernel_size=(5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(4, 4)))
    
    model.add(Conv2D(filters=4, kernel_size=(5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=4, kernel_size=(5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(4, 4)))
    
    model.add(Flatten())
    
    model.add(Dense(10, activation='relu'))
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
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(resize_width, resize_width, n_channel)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=metric_list)
    
    return model