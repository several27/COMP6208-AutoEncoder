import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, UpSampling3D
from keras.models import Sequential, Model
from keras import backend as K
import os.path
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.models import load_model
import h5py


def basic_cnn_upscaler_mnist(x_train, y_train, x_test, y_test, epochs=5,
                            retrain=False, model_filename="basic_cnn_upscaler_mnist.h5"):
    if os.path.isfile(model_filename) and not retrain:
        upscaler = load_model(model_filename)
        upscaler.summary()
    else:
        input_image_dim= np.shape(x_train[0])

        upscaler = Sequential([
            Conv2D(14, (3, 3), activation='relu', padding='same', input_shape=input_image_dim), 
            UpSampling2D((2, 2)),
            Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])
        
        upscaler.compile(optimizer='adadelta', loss='mean_squared_error')
        upscaler.summary()
        upscaler.fit(x_train, y_train,
                          epochs=epochs,
                          batch_size=128,
                          shuffle=True,
                          validation_data=(x_test, y_test))
        upscaler.save(model_filename)
    return upscaler


def basic_cnn_downscaler_mnist(x_train, y_train, x_test, y_test, epochs=5,
                            retrain=False, model_filename="basic_cnn_downscaler_mnist.h5"):
    if os.path.isfile(model_filename) and not retrain:
        upscaler = load_model(model_filename)
        upscaler.summary()
    else:
        input_image_dim= np.shape(x_train[0])
        output_image_dim = np.shape(y_train[0])
        print(input_image_dim, output_image_dim)
        upscaler = Sequential([
            Conv2D(14, (3, 3), activation='relu', padding='same', input_shape=input_image_dim), 
            MaxPooling2D(pool_size=2),
            Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
        ])

        upscaler.compile(optimizer='adadelta', loss='mean_squared_error')
        upscaler.summary()
        
        upscaler.fit(x_train, y_train,
                          epochs=epochs,
                          batch_size=128,
                          shuffle=True,
                          validation_data=(x_test, y_test))
        upscaler.save(model_filename)
    return upscaler


def basic_dense_upscaler_mnist(x_train, y_train, x_test, y_test, epochs=5,
                            retrain=False, model_filename="basic_dense_upscaler_mnist.h5"):
    if os.path.isfile(model_filename) and not retrain:
        upscaler = load_model(model_filename)
        upscaler.summary()
    else:
        input_image_dim = np.shape(x_train[0])
        output_image_dim = np.shape(y_train[0])

        upscaler = Sequential([
            Flatten(input_shape=input_image_dim),
            Dense((392), activation='relu'),
            Dense((784), activation='sigmoid'),
            Reshape(target_shape = output_image_dim),
        ])

        upscaler.compile(optimizer='adadelta', loss='mean_squared_error')
        upscaler.summary()
        
        upscaler.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, y_test))
        upscaler.save(model_filename)

    return upscaler


def basic_dense_downscaler_mnist(x_train, y_train, x_test, y_test, epochs=5,
                            retrain=False, model_filename="basic_dense_donwscaler_mnist.h5"):
    if os.path.isfile(model_filename) and not retrain:
        upscaler = load_model(model_filename)
        upscaler.summary()
    else:
        input_image_dim = np.shape(x_train[0])
        output_image_dim = np.shape(y_train[0])

        upscaler = Sequential([
            Flatten(input_shape=input_image_dim),
            Dense((392), activation='relu'),
            Dense((196), activation='sigmoid'),
            Reshape(target_shape = output_image_dim),
        ])
        
        upscaler.compile(optimizer='adadelta', loss='mean_squared_error')
        upscaler.summary()
        
        upscaler.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, y_test))
        upscaler.save(model_filename)

    return upscaler


def basic_cnn_autoencoder_mnist(x_train, y_train, x_test, y_test, epochs=5,
                            retrain=False, model_filename="basic_cnn_autoencoder_mnist.h5"):
    if os.path.isfile(model_filename) and not retrain:
        upscaler = load_model(model_filename)
        upscaler.summary()
    else:
        input_image_dim= np.shape(x_train[0])
        output_image_dim = np.shape(y_train[0])
        print(input_image_dim, output_image_dim)
        upscaler = Sequential([
            Conv2D(14, (3, 3), activation='relu', padding='same', input_shape=input_image_dim), 
            MaxPooling2D(pool_size=2),
            Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
            Conv2D(14, (3, 3), activation='relu', padding='same', input_shape=input_image_dim), 
            UpSampling2D((2, 2)),
            Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])

        upscaler.compile(optimizer='adadelta', loss='mean_squared_error')
        upscaler.summary()
        
        upscaler.fit(x_train, y_train,
                     epochs=epochs,
                     batch_size=128,
                     shuffle=True,
                     validation_data=(x_test, y_test))
        upscaler.save(model_filename)
    return upscaler


def basic_dense_autoencoder_mnist(x_train, y_train, x_test, y_test, epochs=5,
                            retrain=False, model_filename="basic_dense_autoencoder_mnist.h5"):
    if os.path.isfile(model_filename) and not retrain:
        upscaler = load_model(model_filename)
        upscaler.summary()
    else:
        input_image_dim= np.shape(x_train[0])
        output_image_dim = np.shape(y_train[0])
        print(input_image_dim, output_image_dim)
        upscaler = Sequential([
            Flatten(input_shape=input_image_dim),
            Dense((392), activation='relu'),
            Dense((196), activation='sigmoid'),
            Dense((392), activation='relu'),
            Dense((784), activation='sigmoid'),
            Reshape(target_shape = output_image_dim),
        ])

        upscaler.compile(optimizer='adadelta', loss='mean_squared_error')
        upscaler.summary()
        
        upscaler.fit(x_train, y_train,
                     epochs=epochs,
                     batch_size=128,
                     shuffle=True,
                     validation_data=(x_test, y_test))
        upscaler.save(model_filename)
    return upscaler



