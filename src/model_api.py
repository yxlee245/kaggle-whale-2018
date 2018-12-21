### Functions for pre-trained models ###
from keras.models import Model

from keras.layers import Dense, Dropout

def getPretrainedModel(model, num_classes, num_layers_freeze):
    '''
    model - One of the following: ResNet50, DenseNet201, MobileNetV2 or NASNetMobile
    num_classes - Number of output classes
    num_layers_freeze - Number of pre-trained layers to freeze,
    counting from the output layer of the pre-trained model
    Returns a compiled model'''
    
    # Freeze layers
    if num_layers_freeze == 0:
        for layer in model.layers:
            layer.trainable = False
    else:
        for layer in model.layers[:-num_layers_freeze]:
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