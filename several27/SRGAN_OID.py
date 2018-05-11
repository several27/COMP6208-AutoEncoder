import os
from multiprocessing.pool import Pool

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import ujson
from PIL import Image
import matplotlib as mpl
from collections import defaultdict
from keras import backend as K
from keras.applications import ResNet50, VGG16
from keras.applications.vgg16 import WEIGHTS_PATH_NO_TOP
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.datasets import cifar10
from keras.engine.topology import Layer
from keras.layers import Input, Conv2D, PReLU, BatchNormalization, Add, LeakyReLU, Dense, Flatten, \
    GlobalAveragePooling2D, Dropout, Convolution2D, MaxPooling2D, Activation
from keras.layers.merge import Concatenate
from keras.losses import binary_crossentropy, mean_squared_error
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import conv_utils, to_categorical, get_file
from keras_contrib.losses import DSSIMObjective
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

mpl.use('Agg')
import matplotlib.pyplot as plt


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


def srgan_generator(input_shape, input_=None, ratio=4):
    kernel_size = (3, 3)
    kernel_size_last = (9, 9)
    features = 64
    features_shuffle = 256
    features_last = 3
    B = 16

    # 1, 2, 3
    input_1 = input_ if input_ is not None else Input(shape=input_shape)
    conv2d_2 = Conv2D(filters=features, kernel_size=kernel_size, strides=(1, 1), padding='same',
                      name='generator_conv2d_2')(input_1)
    prelu_3 = PReLU(name='generator_prelu_3')(conv2d_2)

    # 4 - residual blocks
    last_layer = prelu_3
    for i in range(B):
        conv2d_4_A = Conv2D(filters=features, kernel_size=kernel_size, strides=(1, 1), padding='same',
                            name='generator_conv2d_4_A_%s' % i)(last_layer)
        bn_4_B = BatchNormalization(name='generator_bn_4_B_%s' % i)(conv2d_4_A)
        prelu_4_C = PReLU(name='generator_prelu_4_C_%s' % i)(bn_4_B)
        conv2d_4_D = Conv2D(filters=features, kernel_size=kernel_size, strides=(1, 1), padding='same',
                            name='generator_conv2d_4_D_%s' % i)(prelu_4_C)
        bn_4_E = BatchNormalization(name='generator_bn_4_E_%s' % i)(conv2d_4_D)
        add_4_F = Add(name='generator_add_4_F_%s' % i)([last_layer, bn_4_E])

        last_layer = add_4_F

    # 5, 6, 7
    conv2d_5 = Conv2D(filters=features, kernel_size=kernel_size, strides=(1, 1), padding='same',
                      name='generator_conv2d_5')(last_layer)
    bn_6 = BatchNormalization(name='generator_bn_6')(conv2d_5)
    add_7 = Add(name='generator_add_7')([prelu_3, bn_6])

    # 8 - shuffle block
    last_layer = add_7
    for i in range(ratio // 2):
        conv2d_8_A = Conv2D(filters=features_shuffle, kernel_size=kernel_size, strides=(1, 1), padding='same',
                            name='generator_conv2d_8_A_%s' % i)(last_layer)
        shuffler_8_B = PixelShuffler(name='generator_shuffler_8_B_%s' % i)(conv2d_8_A)
        prelu_8_C = PReLU(name='generator_prelu_8_C_%s' % i)(shuffler_8_B)

        last_layer = prelu_8_C

    # 9
    conv2d_9 = Conv2D(filters=features_last, kernel_size=kernel_size_last, strides=(1, 1),
                      padding='same', activation='tanh', name='generator_conv2d_9')(last_layer)

    return Model(input_1, conv2d_9)


def count_images(path, recursive=False):
    if recursive:
        return sum([1 for dir in os.listdir(path) for file in os.listdir(path + dir) if file.endswith('.jpg')])

    return sum([1 for file in os.listdir(path) if file.endswith('.jpg')])


def open_resized_image(path, size):
    img = Image.open(path)
    width, height = img.size
    width_new, height_new = size

    img_new = Image.new('RGB', (width_new, height_new), (0, 0, 0))
    if height > width:
        img.rotate(90)
        width, height = img.size

    if width > width_new or height > height_new:
        img.thumbnail((width_new, height_new), Image.ANTIALIAS)
        width, height = img.size

    img_new.paste(img, ((width_new - width) // 2, (height_new - height) // 2))

    return img_new


def generator_images(path, size=(1024, 768, 3), ratio=2, batch_size=32, discriminator=False,
                     model_generator=None):
    lr_height, lr_width = size[0] // ratio, size[1] // ratio

    batch_i = 0
    batch = np.zeros((batch_size, size[0], size[1], size[2]))
    batch_scaled = np.zeros((batch_size, lr_height, lr_width, size[2]))

    while True:
        for file in os.listdir(path):
            if not file.endswith('.jpg'):
                continue

            if batch_i == batch_size:
                if discriminator is True and model_generator is not None:
                    split = max(len(batch) // 2, 1)
                    batch_1, batch_2 = batch[:split], batch[split:]
                    batch_predicted = np.abs(model_generator.predict(batch_scaled))
                    batch_predicted_1, batch_predicted_2 = batch_predicted[:split], batch_predicted[split:]

                    yield [np.concatenate([batch_predicted_1, batch_1]),
                           np.concatenate([batch_1, batch_1])], \
                          np.concatenate([np.zeros(batch_1.shape[0]), np.ones(batch_1.shape[0])])
                    yield [np.concatenate([batch_predicted_2, batch_2]),
                           np.concatenate([batch_2, batch_2])], \
                          np.concatenate([np.zeros(batch_2.shape[0]), np.ones(batch_2.shape[0])])
                else:
                    yield batch_scaled, batch

                batch_i = 0
                batch = np.zeros((batch_size, size[0], size[1], size[2]))
                batch_scaled = np.zeros((batch_size, lr_height, lr_width, size[2]))

            file_path = path + file

            img = np.array(open_resized_image(file_path, (size[0], size[1]))) / 255
            if len(img.shape) == 2:
                img = np.asarray(np.dstack((img, img, img)))
            img = np.transpose(img, (1, 0, 2))

            batch[batch_i] = img
            batch_scaled[batch_i] = resize(img, (lr_height, lr_width))

            batch_i += 1


def generator_images_discriminator(path, size=(1024, 768, 3), ratio=2, batch_size=32, model_generator=None):
    lr_width, lr_height = size[0] // ratio, size[1] // ratio

    batch_size_predict_scaler = 10
    batch_size_predict = batch_size * batch_size_predict_scaler

    batch_i = 0
    batch = np.zeros((batch_size_predict, size[0], size[1], size[2]))
    batch_scaled = np.zeros((batch_size_predict, lr_width, lr_height, size[2]))

    def _chunks(*args, n_chunks):
        return list(zip(*tuple([np.split(arg, n_chunks) for arg in args])))

    while True:
        for file in os.listdir(path):
            if not file.endswith('.jpg'):
                continue

            if batch_i == batch_size_predict:
                batch_predicted = np.abs(model_generator.predict(batch_scaled))

                for batch_chunk, batch_predicted_chunk in _chunks(batch, batch_predicted,
                                                                  n_chunks=batch_size_predict_scaler):
                    split = max(len(batch_chunk) // 2, 1)
                    batch_1, batch_2 = batch_chunk[:split], batch_chunk[split:]

                    batch_predicted_1, batch_predicted_2 = batch_predicted_chunk[:split], batch_predicted_chunk[split:]

                    X_0, _, X_1, _, y, _ = train_test_split(np.concatenate([batch_predicted_1, batch_1]),
                                                            np.concatenate([batch_1, batch_1]),
                                                            np.concatenate([np.zeros(batch_1.shape[0]),
                                                                            np.ones(batch_1.shape[0])]),
                                                            test_size=0.0)
                    yield [X_0, X_1], y

                    X_0, _, X_1, _, y, _ = train_test_split(np.concatenate([batch_predicted_2, batch_2]),
                                                            np.concatenate([batch_2, batch_2]),
                                                            np.concatenate([np.zeros(batch_2.shape[0]),
                                                                            np.ones(batch_2.shape[0])]),
                                                            test_size=0.0)
                    yield [X_0, X_1], y

                batch_i = 0
                batch = np.zeros((batch_size_predict, size[0], size[1], size[2]))
                batch_scaled = np.zeros((batch_size_predict, lr_width, lr_height, size[2]))

            file_path = path + file

            img = np.array(open_resized_image(file_path, (size[0], size[1]))) / 255
            if len(img.shape) == 2:
                img = np.asarray(np.dstack((img, img, img)))
            img = np.transpose(img, (1, 0, 2))

            batch[batch_i] = img
            batch_scaled[batch_i] = resize(img, (lr_width, lr_height))

            batch_i += 1


def srgan_discriminator(input_prediction, input_original):
    features_1 = 64
    features_2, features_3, features_4 = 128, 256, 512
    kernel_size = 3, 3
    strides = 1, 1
    strides_2 = 2, 2

    input_1 = Concatenate()([input_prediction, input_original])

    conv2d_2 = Conv2D(filters=features_1, kernel_size=kernel_size, strides=strides, padding='same',
                      name='discriminator_conv2d_2')(input_1)
    lrelu_3 = LeakyReLU(name='discriminator_lrelu_3')(conv2d_2)

    conv2d_4_A = Conv2D(filters=features_1, kernel_size=kernel_size, strides=strides_2, padding='same',
                        name='discriminator_conv2d_4_A')(lrelu_3)
    conv2d_4_B = BatchNormalization(name='discriminator_conv2d_4_B')(conv2d_4_A)
    lrelu_4_c = LeakyReLU(name='discriminator_lrelu_4_c')(conv2d_4_B)

    i = 0
    last_layer = lrelu_4_c
    for _features in [features_2, features_3, features_4]:
        for j in range(2):
            conv2d_5_A = Conv2D(filters=_features, kernel_size=kernel_size, strides=(strides if j == 0 else strides_2),
                                padding='same', name='discriminator_conv2d_5_A_%s' % i)(last_layer)
            conv2d_5_B = BatchNormalization(name='discriminator_conv2d_5_B_%s' % i)(conv2d_5_A)
            lrelu_5_C = LeakyReLU(name='discriminator_lrelu_5_C_%s' % i)(conv2d_5_B)

            last_layer = lrelu_5_C
            i += 1

    # this used to be a Flatten layer but it did not work :(
    flatten_8 = GlobalAveragePooling2D(name='discriminator_flatten_8')(last_layer)
    dense_8 = Dense(1024, name='discriminator_dense_8')(flatten_8)
    lrelu_9 = LeakyReLU(name='discriminator_lrelu_9')(dense_8)
    dense_10 = Dense(1, activation='sigmoid', name='discriminator_dense_10')(lrelu_9)

    return Model([input_prediction, input_original], dense_10)


def train_generator(path_train, path_val, train_version, epochs, batch_size, dimensions, ratio):
    dimensions_small = dimensions[0] // ratio, dimensions[1] // ratio, dimensions[2]

    model_generator = srgan_generator(dimensions_small)
    model_generator.compile(loss=DSSIMObjective(), optimizer='adam', metrics=['accuracy'])
    model_generator.summary()

    checkpointer = ModelCheckpoint(
        filepath='data/srgan_generator_weights_%s.{epoch:03d}_{val_acc:.4f}.hdf5' % train_version,
        verbose=1, save_best_only=False)
    tb_callback = TensorBoard(log_dir='data/tensorboard/', histogram_freq=0, write_graph=True, write_images=True)

    n_train = 25000  # count_images(path_train)
    n_val = 1000  # count_images(path_val)
    with tf.device('/gpu:0'):
        pass
        # model_generator.fit_generator(
        #     generator_images_classification_places365(path_train, dimensions, batch_size=batch_size, ratio=ratio,
        #                                               limit=n_train, net='generator'),
        #     steps_per_epoch=n_train // batch_size,
        #     validation_data=generator_images_classification_places365(path_val, dimensions, batch_size=batch_size,
        #                                                               ratio=ratio,
        #                                                               limit=n_val, net='generator'),
        #     validation_steps=n_val // batch_size, epochs=epochs,
        #     callbacks=[checkpointer, tb_callback])

        # model_generator.fit_generator(generator_images(path_train, dimensions, batch_size=batch_size, ratio=ratio),
        #                               steps_per_epoch=n_train // batch_size,
        #                               validation_data=generator_images(path_val, dimensions,
        #                                                                batch_size=batch_size, ratio=ratio),
        #                               validation_steps=n_val // batch_size, epochs=epochs,
        #                               callbacks=[checkpointer, tb_callback])

    model_generator.load_weights('data/srgan_generator_weights_11.018_0.9207.hdf5')

    plt.figure(figsize=(20, 80))
    img_gen = generator_images_classification_places365(path_val, dimensions, batch_size=batch_size, ratio=ratio,
                                                        limit=n_val, net='generator')
    for x_test_batch in img_gen:
        x_test_predicted = np.abs(model_generator.predict(x_test_batch[0]))
        for i, x_test_img in enumerate(x_test_batch[1][:10]):
            plt.subplot(10, 3, (i * 3) + 1)
            plt.imshow(x_test_batch[0][i])

            plt.subplot(10, 3, (i * 3) + 2)
            plt.imshow(x_test_predicted[i])

            plt.subplot(10, 3, (i * 3) + 3)
            plt.imshow(x_test_img)

        break

    plt.savefig('data/srgan_generator_places365_results_%s.png' % (train_version))


def train_discriminator(path_train, path_val, train_version, epochs, batch_size, dimensions, ratio):
    dimensions_small = dimensions[0] // ratio, dimensions[1] // ratio, dimensions[2]

    model_generator = srgan_generator(dimensions_small)
    model_generator.compile(optimizer='adam', loss='mse')

    discriminator_input_prediction = Input(shape=dimensions)
    discriminator_input_original = Input(shape=dimensions)
    model_discriminator = srgan_discriminator(discriminator_input_prediction, discriminator_input_original)

    model_discriminator.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model_discriminator.summary()

    model_generator.load_weights('data/srgan_generator_weights_11.018_0.9207.hdf5')

    checkpointer = ModelCheckpoint(
        filepath='data/srgan_discriminator_weights_%s.{epoch:03d}_{val_acc:.4f}.hdf5' % train_version,
        verbose=1, save_best_only=False)
    tb_callback = TensorBoard(log_dir='data/tensorboard/', histogram_freq=0, write_graph=True, write_images=True)

    n_train = 1000  # count_images(path_train)
    n_val = 100  # count_images(path_val)

    metrics_names = model_discriminator.metrics_names
    with tf.device('/gpu:0'):
        for epoch in range(epochs):
            print()
            print('Epoch %s / %s' % (epoch, epochs))

            with tqdm(total=n_train) as progress:
                for X, y in generator_images_classification_places365(path_train, dimensions, batch_size=batch_size,
                                                                      ratio=ratio, net='discriminator',
                                                                      model_generator=model_generator, limit=n_train):
                    metrics = model_discriminator.train_on_batch(X, y)
                    progress.set_description('Discriminator: %s: %s; %s: %s;' %
                                             (metrics_names[0], metrics[0], metrics_names[1], metrics[1]))
                    progress.update()

                    if progress.n >= n_train:
                        break

                eval_metrics = model_discriminator.evaluate_generator(
                    generator_images_classification_places365(path_val, dimensions, batch_size=batch_size,
                                                              ratio=ratio, net='discriminator',
                                                              model_generator=model_generator,
                                                              limit=n_val),
                    (n_val // batch_size) * 2)
                print('Model discriminator evaluation: %s: %s; %s: %s;' % (metrics_names[0], eval_metrics[0],
                                                                           metrics_names[1], eval_metrics[1]))

        model_discriminator.save_weights('data/srgan_discriminator_%s.hdf5' % train_version)


def srgan_loss(_generator_output, _discriminator_in_original):
    def _srgan_loss(y_true, y_pred):
        # return binary_crossentropy(y_true, y_pred) + \
        #        K.sum(mean_squared_error(_generator_output, _discriminator_in_original), axis=(-1, -2))

        return binary_crossentropy(y_true, y_pred) + DSSIMObjective()(_generator_output, _discriminator_in_original)

    return _srgan_loss


def set_trainable(model, key_word, value=True):
    layers_list = [layer for layer in model.layers if key_word in layer.name]
    for layer in layers_list:
        layer.trainable = value


def train_srgan(path_train, path_val, train_version, epochs, batch_size, dimensions, ratio):
    dimensions_small = dimensions[0] // ratio, dimensions[1] // ratio, dimensions[2]

    generator_in = Input(shape=dimensions_small)
    model_generator = srgan_generator(dimensions_small, input_=generator_in)

    discriminator_in_predicted = Input(shape=dimensions)
    discriminator_in_original = Input(shape=dimensions)
    model_discriminator = srgan_discriminator(discriminator_in_predicted, discriminator_in_original)

    generator_out = model_generator(generator_in)
    discriminator_out = model_discriminator([generator_out, discriminator_in_original])
    model_srgan = Model([generator_in, discriminator_in_original], outputs=discriminator_out)

    model_generator.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model_discriminator.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model_srgan.compile(optimizer=Adam(), loss=srgan_loss(generator_out, discriminator_in_original),
                        metrics=['accuracy'])

    model_generator.load_weights('data/srgan_generator_weights_11.018_0.9207.hdf5')
    model_discriminator.load_weights('data/srgan_discriminator_10.hdf5')

    checkpointer = ModelCheckpoint(
        filepath='data/srgan_srgan_weights_%s.{epoch:03d}_{val_acc:.4f}.hdf5' % train_version,
        verbose=1, save_best_only=False)
    tb_callback = TensorBoard(log_dir='data/tensorboard/', histogram_freq=0, write_graph=True, write_images=True)

    n_train = 10 * 1000  # count_images(path_train)
    n_val = 1000  # count_images(path_val)

    metrics_names = model_discriminator.metrics_names
    with tf.device('/gpu:0'):
        for epoch in range(epochs):
            print()
            print('Epoch %s / %s' % (epoch, epochs))

            total = (n_train // batch_size) * 2
            with tqdm(total=total) as progress:
                for X, y in generator_images_classification_places365(path_train, dimensions, batch_size=batch_size,
                                                                      ratio=ratio, limit=n_train, net='generator'):
                    x_batch_scaled, x_batch = X

                    set_trainable(model_generator, 'generator', True)
                    set_trainable(model_discriminator, 'discriminator', False)
                    metrics_srgan = model_srgan.train_on_batch([x_batch_scaled, x_batch], [1] * batch_size)

                    set_trainable(model_generator, 'generator', False)
                    set_trainable(model_discriminator, 'discriminator', True)

                    x_predicted_batch = model_generator.predict(x_batch_scaled)
                    x1_disc = np.concatenate([x_predicted_batch, x_batch])
                    x2_disc = np.concatenate([x_batch, x_batch])
                    y_disc = np.concatenate([np.zeros(batch_size), np.ones(batch_size)])

                    metrics_discriminator = model_discriminator.train_on_batch([x1_disc, x2_disc], y_disc)

                    progress.set_description('Srgan: %s: %s; %s: %s; Discriminator: %s: %s; %s: %s' %
                                             (metrics_names[0], metrics_srgan[0], metrics_names[1], metrics_srgan[1],
                                              metrics_names[0], metrics_discriminator[0], metrics_names[1],
                                              metrics_discriminator[1]))
                    progress.update()

                    if progress.n % 100 == 0:
                        model_srgan.save_weights('data/srgan_progress/srgan_%s_%s.hdf5' % (train_version, progress.n))
                        model_generator.save_weights('data/srgan_progress/srgan_g_%s_%s.hdf5' %
                                                     (train_version, progress.n))

                        gen = generator_images_classification_places365(path_val, dimensions, batch_size=batch_size,
                                                                        ratio=ratio, limit=n_val, net='generator')

                        plt.figure(figsize=(20, 80))
                        for p_batch_scaled, p_batch in gen:
                            p_batch_predicted = np.abs(model_generator.predict(p_batch_scaled))
                            for i, p_img_scaled in enumerate(p_batch_scaled[:10]):
                                p_img = p_batch[i]
                                p_img_predicted = p_batch_predicted[i]

                                plt.subplot(10, 3, (i * 3) + 1)
                                plt.imshow(p_img_scaled)

                                plt.subplot(10, 3, (i * 3) + 2)
                                plt.imshow(p_img_predicted)

                                plt.subplot(10, 3, (i * 3) + 3)
                                plt.imshow(p_img)

                            break

                        plt.savefig('data/srgan_progress/results_%s_%s_%s.png' % (train_version, epoch, progress.n))

                    if progress.n >= total:
                        break

                eval_g = model_generator.evaluate_generator(
                    generator_images_classification_places365(path_val, dimensions, batch_size=batch_size,
                                                              ratio=ratio, limit=n_val, net='generator'),
                    steps=n_val // batch_size)
                print('Model generator evaluation: %s: %s; %s: %s' % (metrics_names[0], eval_g[0], metrics_names[1],
                                                                      eval_g[1]))

        model_srgan.save_weights('data/srgan_%s.hdf5' % train_version)
        model_generator.save_weights('data/srgan_g_%s.hdf5' % train_version)


def generator_images_classification(path_dataset, path_classes, path_classes_descs, size=(1024, 768, 3), batch_size=32):
    df_classes = pd.read_csv(path_classes)
    df_classes_descs = pd.read_csv(path_classes_descs, header=None)

    classes = {}
    for i, row in enumerate(df_classes_descs.itertuples()):
        classes[row._1] = i

    image_id_classes = defaultdict(list)
    for row in df_classes.itertuples():
        image_id_classes[row.ImageID].append(classes[row.LabelName])

    size_classes = df_classes_descs.shape[0]

    batch_i = 0
    batch = np.zeros((batch_size, size[0], size[1], size[2]))
    batch_classes = np.zeros((batch_size, size_classes))

    while True:
        for file in os.listdir(path_dataset):
            if not file.endswith('.jpg'):
                continue

            if batch_i == batch_size:
                yield batch, batch_classes

                batch_i = 0
                batch = np.zeros((batch_size, size[0], size[1], size[2]))
                batch_classes = np.zeros((batch_size, size_classes))

            file_path = path_dataset + file

            img = np.array(open_resized_image(file_path, (size[0], size[1]))) / 255
            if len(img.shape) == 2:
                img = np.asarray(np.dstack((img, img, img)))
            img = np.transpose(img, (1, 0, 2))

            batch[batch_i] = img
            batch_classes[batch_i] = [(1 if i in image_id_classes[file[:-4]] else 0) for i in range(0, size_classes)]
            batch_i += 1


def chunks(l, batch_size, size, classes, size_classes, lr_width, lr_height, net, model_generator):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), batch_size):
        yield l[i:i + batch_size], batch_size, size, classes, size_classes, lr_width, lr_height, net, model_generator


def _generator_images_classification_places365(params):
    dir_files, batch_size, size, classes, size_classes, lr_width, lr_height, net, model_generator = params

    batch_i = 0
    batch = np.zeros((batch_size, size[0], size[1], size[2]))
    batch_classes = np.zeros((batch_size, size_classes))
    batch_scaled = np.zeros((batch_size, lr_width, lr_height, size[2]))

    for dir, path_dir, file in dir_files:
        file_path = path_dir + file

        img = np.array(open_resized_image(file_path, (size[0], size[1]))) / 255
        if len(img.shape) == 2:
            img = np.asarray(np.dstack((img, img, img)))
        img = np.transpose(img, (1, 0, 2))

        batch[batch_i] = img
        batch_classes[batch_i] = np.zeros(size_classes)
        batch_classes[batch_i, classes[dir]] = 1
        batch_scaled[batch_i] = resize(img, (lr_height, lr_width))
        batch_i += 1

    if batch_i < batch_size:
        return None

    if net == 'classification':
        return batch, batch_classes
    elif net == 'generator':
        return batch_scaled, batch
    elif net == 'discriminator':
        batch_predicted = model_generator.predict(batch_scaled)

        return [
                   np.concatenate([batch_predicted, batch]),
                   np.concatenate([batch, batch]),
               ], np.concatenate([np.zeros(batch.shape[0]), np.ones(batch.shape[0])])


def generator_images_classification_places365(path_dataset, size=(256, 256, 3), batch_size=32, ratio=4, limit=None,
                                              net='classification', model_generator=None):
    classes = dict([(d, i) for i, d in enumerate(os.listdir(path_dataset))])
    size_classes = len(classes)

    lr_width, lr_height = size[0] // ratio, size[1] // ratio

    path_files_list = 'data/places365_%s.json' % limit
    if os.path.isfile(path_files_list):
        with open(path_files_list, 'r') as _in:
            dir_files = ujson.load(_in)
    else:
        dir_files = []
        for dir in os.listdir(path_dataset):
            path_dir = path_dataset + dir + '/'
            for file in os.listdir(path_dir):
                if not file.endswith('.jpg'):
                    continue

                dir_files.append((dir, path_dir, file))

        dir_files = shuffle(dir_files, random_state=42)
        if limit is not None:
            dir_files = dir_files[:limit]

        with open(path_files_list, 'w') as _out:
            ujson.dump(dir_files, _out)

    while True:
        chunks_iter = chunks(dir_files, batch_size, size, classes, size_classes, lr_width, lr_height, net,
                             model_generator)
        for data in chunks_iter:
            data = _generator_images_classification_places365(data)

            if data is None:
                continue

            yield data


def srgan_vgg(input_shape, classes):
    # Append the initial inputs to the outputs of the SRResNet
    x_in = Input(input_shape)

    # Begin adding the VGG layers
    x = Conv2D(64, 3, 3, activation='relu', name='vgg_conv1_1', border_mode='same')(x_in)

    x = Conv2D(64, 3, 3, activation='relu', name='vgg_conv1_2', border_mode='same')(x)
    x = MaxPooling2D(name='vgg_maxpool1')(x)

    x = Conv2D(128, 3, 3, activation='relu', name='vgg_conv2_1', border_mode='same')(x)

    x = Conv2D(128, 3, 3, activation='relu', name='vgg_conv2_2', border_mode='same')(x)
    x = MaxPooling2D(name='vgg_maxpool2')(x)

    x = Conv2D(256, 3, 3, activation='relu', name='vgg_conv3_1', border_mode='same')(x)
    x = Conv2D(256, 3, 3, activation='relu', name='vgg_conv3_2', border_mode='same')(x)

    x = Conv2D(256, 3, 3, activation='relu', name='vgg_conv3_3', border_mode='same')(x)
    x = MaxPooling2D(name='vgg_maxpool3')(x)

    x = Conv2D(512, 3, 3, activation='relu', name='vgg_conv4_1', border_mode='same')(x)
    x = Conv2D(512, 3, 3, activation='relu', name='vgg_conv4_2', border_mode='same')(x)

    x = Conv2D(512, 3, 3, activation='relu', name='vgg_conv4_3', border_mode='same')(x)
    x = MaxPooling2D(name='vgg_maxpool4')(x)

    x = Conv2D(512, 3, 3, activation='relu', name='vgg_conv5_1', border_mode='same')(x)
    x = Conv2D(512, 3, 3, activation='relu', name='vgg_conv5_2', border_mode='same')(x)

    x = Convolution2D(512, 3, 3, activation='relu', name='vgg_conv5_3', border_mode='same')(x)
    x = MaxPooling2D(name='vgg_maxpool5')(x)

    return Model(x_in, x)


def train_vgg(path_train, path_val, train_version, epochs, batch_size, dimensions, ratio):
    checkpointer = ModelCheckpoint(filepath='data/vgg_weights_%s.{epoch:03d}_{val_acc:.4f}.hdf5' % train_version,
                                   verbose=1, save_best_only=False)
    tb_callback = TensorBoard(log_dir='data/tensorboard/', histogram_freq=0, write_graph=True, write_images=True)

    classes = 365

    n_train = 1000 * 1000  # count_images(path_train, recursive=True)
    n_val = 1000  # count_images(path_val, recursive=True)
    with tf.device('/gpu:0'):
        base_model = srgan_vgg(input_shape=dimensions, classes=classes)

        # weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        #                         WEIGHTS_PATH_NO_TOP,
        #                         cache_subdir='models',
        #                         file_hash='6d6bbae143d832006294945121d1f1fc')
        # base_model.load_weights(weights_path)

        # for l in base_model.layers:
        #     l.trainable = False

        top_model = base_model.output
        top_model = Flatten()(top_model)
        top_model = Dense(512, activation='relu')(top_model)
        top_model = BatchNormalization()(top_model)

        top_model = Dropout(0.5)(top_model)
        top_model = Dense(classes, activation='softmax')(top_model)

        model = Model(base_model.input, top_model)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        model.load_weights('data/vgg_pretrained_200k/vgg_weights_11.007_0.2135.hdf5')

        model.fit_generator(
            generator_images_classification_places365(path_train, dimensions, batch_size, limit=n_train),
            steps_per_epoch=n_train // batch_size,
            validation_data=generator_images_classification_places365(path_val, dimensions, batch_size,
                                                                      limit=n_val),
            validation_steps=n_val // batch_size, epochs=epochs, callbacks=[checkpointer, tb_callback])

        model.save_weights('data/vgg_%s.hdf5' % train_version)


def train_vgg_cifar(train_version, epochs, batch_size, dimensions, ratio):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    checkpointer = ModelCheckpoint(filepath='data/vgg_cifar_weights_%s.{epoch:03d}_{val_acc:.4f}.hdf5' % train_version,
                                   verbose=1, save_best_only=False)
    tb_callback = TensorBoard(log_dir='data/tensorboard/', histogram_freq=0, write_graph=True, write_images=True)

    with tf.device('/gpu:0'):
        base_model = srgan_vgg(input_shape=dimensions, classes=num_classes)

        weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                file_hash='6d6bbae143d832006294945121d1f1fc')
        base_model.load_weights(weights_path)

        # for l in base_model.layers:
        #     l.trainable = False

        top_model = base_model.output
        top_model = Flatten()(top_model)
        top_model = Dense(512, activation='relu')(top_model)
        top_model = BatchNormalization()(top_model)

        top_model = Dropout(0.5)(top_model)
        top_model = Dense(num_classes, activation='softmax')(top_model)

        model = Model(base_model.input, top_model)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        model.load_weights('data/vgg_cifar_weights_12.010_0.8011.hdf5')

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[checkpointer, tb_callback],
                  validation_data=(x_test, y_test), verbose=1)
        model.save_weights('data/vgg_cifar_%s.hdf5' % train_version)


def train_generator_cifar(train_version, epochs, batch_size, dimensions, ratio):
    dimensions_small = dimensions[0] // ratio, dimensions[1] // ratio, dimensions[2]

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    num_classes = 10

    x_train_scaled = np.asarray([resize(img, dimensions_small) for img in x_train])
    x_test_scaled = np.asarray([resize(img, dimensions_small) for img in x_test])

    model_generator = srgan_generator(dimensions_small, ratio=ratio)
    model_generator.compile(loss=DSSIMObjective(), optimizer='adam', metrics=['accuracy'])
    model_generator.summary()

    checkpointer = ModelCheckpoint(
        filepath='data/srgan_generator_cifar_dssim_weights_%s.{epoch:03d}_{val_acc:.4f}.hdf5' % train_version,
        verbose=1, save_best_only=False)
    tb_callback = TensorBoard(log_dir='data/tensorboard/srgan_generator_cifar_%s/', histogram_freq=0, write_graph=True,
                              write_images=True)

    # with tf.device('/gpu:0'):
    #     model_generator.fit(x_train_scaled, x_train, batch_size=batch_size, epochs=epochs,
    #                         callbacks=[checkpointer, tb_callback], validation_data=(x_test_scaled, x_test),
    #                         verbose=1)
    #     model_generator.save_weights('data/srgan_generator_cifar_dssim_%s.hdf5' % train_version)

    model_generator.load_weights('data/srgan_generator_cifar_dssim_weights_12.005_0.8808.hdf5')

    plt.figure(figsize=(20, 80))
    x_test_predicted = np.abs(model_generator.predict(x_test_scaled[:10]))
    for i, x_test_img in enumerate(x_test[:10]):
        plt.subplot(10, 3, (i * 3) + 1)
        plt.imshow(x_test_scaled[i])

        plt.subplot(10, 3, (i * 3) + 2)
        plt.imshow(x_test_predicted[i])

        plt.subplot(10, 3, (i * 3) + 3)
        plt.imshow(x_test_img)

    plt.savefig('data/srgan_generator_cifar_dssim_results_%s.png' % (train_version))


def train_discriminator_cifar(train_version, epochs, batch_size, dimensions, ratio):
    dimensions_small = dimensions[0] // ratio, dimensions[1] // ratio, dimensions[2]

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    num_classes = 10

    x_train_scaled = np.asarray([resize(img, dimensions_small) for img in x_train])
    x_test_scaled = np.asarray([resize(img, dimensions_small) for img in x_test])

    model_generator = srgan_generator(dimensions_small, ratio=ratio)
    model_generator.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model_generator.summary()
    model_generator.load_weights('data/srgan_generator_cifar_dssim_weights_12.005_0.8808.hdf5')

    x_train_pred = np.concatenate([x_train, model_generator.predict(x_train_scaled)])
    x_test_pred = np.concatenate([x_test, model_generator.predict(x_test_scaled)])

    y_train = np.concatenate([np.ones(x_train.shape[0]), np.zeros(x_train.shape[0])])
    y_test = np.concatenate([np.ones(x_test.shape[0]), np.zeros(x_test.shape[0])])

    x_train = np.concatenate([x_train, x_train])
    x_test = np.concatenate([x_test, x_test])

    x_train, x_train_pred, y_train = shuffle(x_train, x_train_pred, y_train)
    x_test, x_test_pred, y_test = shuffle(x_test, x_test_pred, y_test)

    discriminator_input_prediction = Input(shape=dimensions)
    discriminator_input_original = Input(shape=dimensions)
    model_discriminator = srgan_discriminator(discriminator_input_prediction, discriminator_input_original)

    model_discriminator.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model_discriminator.summary()

    checkpointer = ModelCheckpoint(
        filepath='data/srgan_discriminator_cifar_dssim_weights_%s.{epoch:03d}_{val_acc:.4f}.hdf5' % train_version,
        verbose=1, save_best_only=False)
    tb_callback = TensorBoard(log_dir='data/tensorboard/srgan_discriminator_cifar_%s/', histogram_freq=0,
                              write_graph=True, write_images=True)

    with tf.device('/gpu:0'):
        model_discriminator.fit([x_train_pred, x_train], y_train, batch_size=batch_size, epochs=epochs,
                                callbacks=[checkpointer, tb_callback], validation_data=([x_test_pred, x_test], y_test),
                                verbose=1)
        model_discriminator.save_weights('data/discriminator_cifar_dssim_%s.hdf5' % train_version)


def train_srgan_cifar(train_version, epochs, batch_size, dimensions, ratio):
    dimensions_small = dimensions[0] // ratio, dimensions[1] // ratio, dimensions[2]

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    num_classes = 10

    x_train_scaled = np.asarray([resize(img, dimensions_small) for img in x_train])
    x_test_scaled = np.asarray([resize(img, dimensions_small) for img in x_test])

    x_train, x_train_scaled = shuffle(x_train, x_train_scaled)
    x_test, x_test_scaled = shuffle(x_test, x_test_scaled)

    generator_in = Input(shape=dimensions_small)
    model_generator = srgan_generator(dimensions_small, input_=generator_in, ratio=ratio)

    discriminator_in_predicted = Input(shape=dimensions)
    discriminator_in_original = Input(shape=dimensions)
    model_discriminator = srgan_discriminator(discriminator_in_predicted, discriminator_in_original)

    generator_out = model_generator(generator_in)
    discriminator_out = model_discriminator([generator_out, discriminator_in_original])
    model_srgan = Model([generator_in, discriminator_in_original], outputs=discriminator_out)

    model_generator.compile(loss=DSSIMObjective(), optimizer='adam', metrics=['accuracy'])
    model_discriminator.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model_srgan.compile(optimizer=Adam(), loss=srgan_loss(generator_out, discriminator_in_original),
                        metrics=['accuracy'])

    model_generator.load_weights('data/srgan_generator_cifar_dssim_weights_12.005_0.8808.hdf5')
    model_discriminator.load_weights('data/srgan_discriminator_cifar_dssim_weights_12.010_0.9999.hdf5')

    def cifar_chunks(l1, l2, batch_size):
        for i in range(0, len(l1), batch_size):
            if len(l1) < i + batch_size:
                break

            yield l1[i:i + batch_size], l2[i:i + batch_size]

    metrics_names = model_discriminator.metrics_names
    with tf.device('/gpu:0'):
        for epoch in range(epochs):
            print()
            print('Epoch %s / %s' % (epoch, epochs))

            total = x_train.shape[0] // batch_size
            with tqdm(total=total) as progress:
                for i, (x_train_batch, x_train_scaled_batch) in enumerate(cifar_chunks(x_train, x_train_scaled,
                                                                                       batch_size)):
                    set_trainable(model_generator, 'generator', True)
                    set_trainable(model_discriminator, 'discriminator', False)
                    metrics_srgan = model_srgan.train_on_batch([x_train_scaled_batch, x_train_batch],
                                                               np.ones(batch_size))

                    x_train_predicted_batch = model_generator.predict(x_train_scaled_batch)
                    x1_disc = np.concatenate([x_train_predicted_batch, x_train_batch])
                    x2_disc = np.concatenate([x_train_batch, x_train_batch])
                    y_disc = np.concatenate([np.zeros(batch_size), np.ones(batch_size)])

                    set_trainable(model_generator, 'generator', False)
                    set_trainable(model_discriminator, 'discriminator', True)
                    metrics_discriminator = model_discriminator.train_on_batch([x1_disc, x2_disc], y_disc)

                    progress.set_description('Srgan: %s: %s; %s: %s; Discriminator: %s: %s; %s: %s' %
                                             (metrics_names[0], metrics_srgan[0], metrics_names[1], metrics_srgan[1],
                                              metrics_names[0], metrics_discriminator[0], metrics_names[1],
                                              metrics_discriminator[1]))
                    progress.update()

                    if progress.n % 100 == 0:
                        model_srgan.save_weights('data/srgan_progress/srgan_%s_%s.hdf5' % (train_version, progress.n))
                        model_generator.save_weights('data/srgan_progress/srgan_g_%s_%s.hdf5' %
                                                     (train_version, progress.n))

                        plt.figure(figsize=(50, 15))
                        x_test_predicted = np.abs(model_generator.predict(x_test_scaled[:10]))
                        for i, x_test_img in enumerate(x_test[:10]):
                            plt.subplot(3, 10, i + 1)
                            plt.imshow(x_test_scaled[i])
                            plt.axis('off')

                            plt.subplot(3, 10, i + 11)
                            plt.imshow(x_test_predicted[i])
                            plt.axis('off')

                            plt.subplot(3, 10, i + 21)
                            plt.imshow(x_test_img)
                            plt.axis('off')

                        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                        plt.savefig('data/srgan_progress/results_%s_%s_%s.png' % (train_version, epoch, progress.n),
                                    bbox_inches='tight')

                    if progress.n >= total:
                        break

                eval_g = model_generator.evaluate(x_test_scaled, x_test)
                print('Model generator evaluation: %s: %s; %s: %s' % (metrics_names[0], eval_g[0], metrics_names[1],
                                                                      eval_g[1]))

        model_srgan.save_weights('data/srgan_cifar_%s.hdf5' % train_version)
        model_generator.save_weights('data/srgan_cifar_g_%s.hdf5' % train_version)


def prepare_data():
    path_data = 'data/'
    path_source = path_data + 'validation/'
    path_destination = path_data + 'validation_preped/'

    dimensions = 512, 384

    with tqdm() as progress:
        for file in os.listdir(path_source):
            if not file.endswith('.jpg'):
                continue

            img_new = open_resized_image(path_source + file, dimensions)
            img_new.save(path_destination + file)
            progress.update()


def main_places365():
    path_data = 'data/'
    path_oid_train = path_data + 'train/'
    path_oid_test = path_data + 'test/'
    path_oid_val = path_data + 'validation/'

    path_oid_resized_train = path_data + 'oid_train_256_192/'
    path_oid_resized_test = path_data + 'oid_test_256_192/'
    path_oid_resized_val = path_data + 'oid_validation_256_192/'

    path_oid_classes = path_data + 'oid_classes/class-descriptions-boxable.csv'
    path_oid_train_classes = path_data + 'oid_classes/train-annotations-human-imagelabels-boxable.csv'
    path_oid_test_classes = path_data + 'oid_classes/test-annotations-human-imagelabels-boxable.csv'
    path_oid_val_classes = path_data + 'oid_classes/validation-annotations-human-imagelabels-boxable.csv'

    path_places36_train = path_data + 'places365_standard/train/'
    path_places36_val = path_data + 'places365_standard/val/'

    train_version = 11
    epochs = 2
    batch_size = 64

    ratio = 4
    # dimensions = 256, 192, 3
    dimensions = 256, 256, 3
    # dimensions = 32, 32, 3

    # train_generator(path_places36_train, path_places36_val, train_version, epochs, batch_size, dimensions, ratio)
    # train_discriminator(path_places36_train, path_places36_val, train_version, epochs, batch_size, dimensions, ratio)
    # train_vgg(path_places36_train, path_places36_val, train_version, epochs, batch_size, dimensions, ratio)
    train_srgan(path_oid_test, path_oid_val, train_version, epochs, batch_size, dimensions, ratio)

    # train_generator_cifar(train_version, epochs, batch_size, dimensions, ratio)
    # train_vgg_cifar(train_version, epochs, batch_size, dimensions, ratio)


def main_cifar():
    # train_version 12 has fixed the divison by 255 !

    train_version = 12
    epochs = 5
    batch_size = 128

    ratio = 2
    # dimensions = 256, 192, 3
    # dimensions = 256, 256, 3
    dimensions = 32, 32, 3

    # train_generator_cifar(train_version, epochs, batch_size, dimensions, ratio)
    # train_discriminator_cifar(train_version, epochs, batch_size, dimensions, ratio)
    train_srgan_cifar(train_version, epochs, batch_size, dimensions, ratio)
    # train_vgg_cifar(train_version, epochs, batch_size, dimensions, ratio)


if __name__ == '__main__':
    # main_places365()
    main_cifar()
