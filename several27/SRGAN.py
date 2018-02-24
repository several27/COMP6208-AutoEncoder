import os

import matplotlib as mpl
mpl.use('Agg')

import numpy as np
from PIL import Image
from tqdm import tqdm 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_float
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.utils import conv_utils
from keras.engine.topology import Layer
from keras.datasets import mnist, cifar10
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, PReLU, BatchNormalization, Add

path_data = '/home/ubuntu/COMP6208-AutoEncoder/several27/data/'
path_open_images_560_420 = path_data + 'open_images_560_420/'
path_open_images_560_420_train = path_data + 'open_images_560_420_train/'
path_open_images_560_420_val = path_data + 'open_images_560_420_val/'


def generator_images(path, size=(420, 560, 3), ratio=2, batch_size=32):
    lr_height, lr_width = size[0] // ratio, size[1] // ratio
    
    batch_i = 0
    batch = np.zeros((batch_size, size[0], size[1], size[2]))
    batch_scaled = np.zeros((batch_size, lr_height, lr_width, size[2]))
    
    while True:
        for file in os.listdir(path):
            if not file.endswith('.jpg'):
                continue 
            
            if batch_i == batch_size:
                yield batch_scaled, batch
                
                batch_i = 0
                batch = np.zeros((batch_size, size[0], size[1], size[2]))
                batch_scaled = np.zeros((batch_size, lr_height, lr_width, size[2]))
            
            file_path = path + file
            img = img_as_float(Image.open(file_path))
            
            batch[batch_i] = img
            batch_scaled[batch_i] = resize(img, (lr_height, lr_width))
            
            batch_i += 1

            
def count_images(path):
    return sum([1 for file in os.listdir(path) if file.endswith('.jpg')])


# from https://gist.github.com/t-ae/6e1016cc188104d123676ccef3264981

class PixelShuffler(Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')

    def call(self, inputs):

        input_shape = K.int_shape(inputs)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_first':
            batch_size, c, h, w = input_shape
            if batch_size is None:
                batch_size = -1
            rh, rw = self.size
            oh, ow = h * rh, w * rw
            oc = c // (rh * rw)

            out = K.reshape(inputs, (batch_size, rh, rw, oc, h, w))
            out = K.permute_dimensions(out, (0, 3, 4, 1, 5, 2))
            out = K.reshape(out, (batch_size, oc, oh, ow))
            return out

        elif self.data_format == 'channels_last':
            batch_size, h, w, c = input_shape
            if batch_size is None:
                batch_size = -1
            rh, rw = self.size
            oh, ow = h * rh, w * rw
            oc = c // (rh * rw)

            out = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
            out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
            out = K.reshape(out, (batch_size, oh, ow, oc))
            return out

    def compute_output_shape(self, input_shape):

        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_first':
            height = input_shape[2] * self.size[0] if input_shape[2] is not None else None
            width = input_shape[3] * self.size[1] if input_shape[3] is not None else None
            channels = input_shape[1] // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[1]:
                raise ValueError('channels of input and size are incompatible')

            return (input_shape[0],
                    channels,
                    height,
                    width)

        elif self.data_format == 'channels_last':
            height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
            width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
            channels = input_shape[3] // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[3]:
                raise ValueError('channels of input and size are incompatible')

            return (input_shape[0],
                    height,
                    width,
                    channels)

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(PixelShuffler, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def sgan_generator(input_shape):
    kernel_size = (3, 3)
    kernel_size_last = (9, 9)
    features = 64
    features_shuffle = 256
    features_last = 3
    B = 16

    # 1, 2, 3
    input_1 = Input(shape=input_shape)
    conv2d_2 = Conv2D(filters=features, kernel_size=kernel_size, strides=(1, 1), padding='same')(input_1)
    prelu_3 = PReLU()(conv2d_2)

    # 4 - residual blocks
    last_layer = prelu_3
    for _ in range(B):
        conv2d_4_A = Conv2D(filters=features, kernel_size=kernel_size, strides=(1, 1), padding='same')(last_layer)
        bn_4_B = BatchNormalization()(conv2d_4_A)
        prelu_4_C = PReLU()(bn_4_B)
        conv2d_4_D = Conv2D(filters=features, kernel_size=kernel_size, strides=(1, 1), padding='same')(prelu_4_C)
        bn_4_E = BatchNormalization()(conv2d_4_D)
        add_4_F = Add()([last_layer, bn_4_E])

        last_layer = add_4_F

    # 5, 6, 7
    conv2d_5 = Conv2D(filters=features, kernel_size=kernel_size, strides=(1, 1), padding='same')(last_layer)
    bn_6 = BatchNormalization()(conv2d_5)
    add_7 = Add()([prelu_3, bn_6])

    # 8 - shuffle block
    last_layer = add_7
    for _ in range(1):
        conv2d_8_A = Conv2D(filters=features_shuffle, kernel_size=kernel_size, strides=(1, 1), padding='same')(last_layer)
        shuffler_8_B = PixelShuffler()(conv2d_8_A)
        prelu_8_C = PReLU()(shuffler_8_B)

        last_layer = prelu_8_C

    # 9 
    conv2d_5 = Conv2D(filters=features_last, kernel_size=kernel_size_last, strides=(1, 1), 
                      padding='same')(last_layer)

    model_generator = Model(input_1, conv2d_5)
    model_generator.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    
    return model_generator


def train():
    train_version = 4
    batch_size = 3
    epochs = 1
    
    n_train = count_images(path_open_images_560_420_train)
    n_val = count_images(path_open_images_560_420_val)
    
    model_generator = sgan_generator((210, 280, 3))
    model_generator.summary()
    
    checkpointer = ModelCheckpoint(filepath='data/srgan_generator_weights_%s.{epoch:03d}_{val_acc:.4f}.hdf5' % train_version, 
                               verbose=1, save_best_only=False)
    tb_callback = TensorBoard(log_dir='data/tensorboard/', histogram_freq=0, write_graph=True, write_images=True)
   
    with tf.device('/gpu:0'):
        model_generator.fit_generator(generator_images(path_open_images_560_420_train, batch_size=batch_size), 
                                      steps_per_epoch=n_train // batch_size,
                                      validation_data=generator_images(path_open_images_560_420_val, 
                                                                       batch_size=batch_size),
                                      validation_steps=n_val // batch_size, epochs=epochs, callbacks=[tb_callback, checkpointer])
        
    model_generator.save('data/srgan_generator_%s.model' % train_version)
    
    for g, type_ in [(generator_images(path_open_images_560_420_train, batch_size=1), 'train'), 
                     (generator_images(path_open_images_560_420_val, batch_size=batch_size), 'test')]:
        plt.figure(type_, figsize=(15, 30))
        for i, (x_lr, x) in enumerate(g):
            x_lr_ = (x_lr[0] * 255).astype(np.uint8)
            x_ = (x[0] * 255).astype(np.uint8)

            plt.subplot(10, 3, (i * 3) + 1)
            plt.imshow(x_lr_)

            plt.subplot(10, 3, (i * 3) + 2)
            plt.imshow(x_)

            plt.subplot(10, 3, (i * 3) + 3)
            plt.imshow((model_generator.predict(x_lr)[0] * 255).astype(np.uint8))

            if i >= 9:
                break

        plt.savefig('data/srgan_generator_%s_%s.png' % (train_version, type_))


if __name__ == '__main__': 
    train()
