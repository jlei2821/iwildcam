# Import Tensorflow Modules
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, InputLayer, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D

def tf_cnn_model(input_shape):

    # Define Parameters
    kernel_size = (3, 3)
    strides = (2, 2)
    pooling = (2, 2)
    
    m = Sequential()

    # For now, using strides of 2 instead of max pooling to dimensionality reduction
    m.add(Conv2D(filters=64, kernel_size=kernel_size, strides=strides, padding='same', input_shape=input_shape))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=128, kernel_size=kernel_size, strides=strides, padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=256, kernel_size=kernel_size, strides=strides, padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=512, kernel_size=kernel_size, strides=strides, padding='same'))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

    m.add(Flatten())

    m.add(BatchNormalization())
    m.add(Dense(128))
    m.add(Activation('relu'))

    m.add(BatchNormalization())
    m.add(Dense(23, activation='softmax', activity_regularizer=l2(0.01)))

    return m



def tf_cnn_model_pool(input_shape):
    
    # Define Parameters
    kernel_size = (3, 3)
    strides = (2, 2)
    pooling = (2, 2)

    m = Sequential()

    # For now, using strides of 2 instead of max pooling to dimensionality reduction
    m.add(Conv2D(filters=64, kernel_size=kernel_size, padding='same', input_shape=input_shape))
    m.add(MaxPooling2D(pooling))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=128, kernel_size=kernel_size, padding='same'))
    m.add(MaxPooling2D(pooling))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=256, kernel_size=kernel_size, padding='same'))
    m.add(MaxPooling2D(pooling))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

    m.add(Conv2D(filters=512, kernel_size=kernel_size, padding='same'))
    m.add(MaxPooling2D(pooling))
    m.add(Activation('relu'))
    m.add(BatchNormalization())

    m.add(Flatten())

    m.add(BatchNormalization())
    m.add(Dense(128))
    m.add(Activation('relu'))

    m.add(BatchNormalization())
    m.add(Dense(23, activation='softmax', activity_regularizer=l2(0.01)))

    return m