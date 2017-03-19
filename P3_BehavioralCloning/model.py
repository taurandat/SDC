import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout, Lambda, Activation, MaxPooling2D, BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adadelta

import generate_utils
import os_utils

tf.python.control_flow_ops = tf

# hyperparameters
number_of_epochs = 20
number_of_samples_per_epoch = 25600
number_of_validation_samples = 6400
activation_layers = ['relu', 'elu']


def get_NVIDIA_model(activation_layer):
    model = Sequential()

    # Preprocess: Data normalization
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

    # Several blocks after the preprocess step
    # The logic of each block is simple: Conv2D - BatchNorm - Activation -
    # MaxPool - Dropout

    # First block
    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation(activation_layer))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.25))

    # Second block
    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation(activation_layer))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.25))

    # Third block
    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation(activation_layer))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.25))

    # Forth block
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation(activation_layer))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.25))

    # Fifth block
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation(activation_layer))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.25))

    # Flatten the final output from all the blocks
    model.add(Flatten())

    # Final four feedforward layers with activation
    model.add(Dense(1164))
    model.add(Activation(activation_layer))
    model.add(Dense(100))
    model.add(Activation(activation_layer))
    model.add(Dense(50))
    model.add(Activation(activation_layer))
    model.add(Dense(10))
    model.add(Activation(activation_layer))

    # Model output: Vehicle control
    model.add(Dense(1))

    model.summary()

    model.compile(optimizer=Adadelta(), loss="mse")

    return model


if __name__ == '__main__':
    for layer in activation_layers:
        model = get_NVIDIA_model()

        # generate data on the fly
        train_gen = generate_utils.generate_next_batch()
        valid_gen = generate_utils.generate_next_batch()

        history = model.fit_generator(train_gen,
                                      samples_per_epoch=number_of_samples_per_epoch,
                                      nb_epoch=number_of_epochs,
                                      validation_data=valid_gen,
                                      nb_val_samples=number_of_validation_samples,
                                      verbose=1)

        os_utils.save_model(model,
                            model_name='model_%s_drop.json' % layer,
                            weights_name='model_%s_drop.h5' % layer)
