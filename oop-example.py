from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse


class DCGAN:

    def __init__(self,
                 model_name='one-dim_GAN',
                 save_interval=500,
                 latent_size=100,
                 batch_size=64,
                 train_steps=40000,
                 lr=2e-4,
                 decay=6e-8,
                 input_shape=(28, 28, 1),
                 lr_coefficient=0.5,
                 decay_coefficient=0.5
                 ):

        self.model_name = model_name
        self.save_interval = save_interval
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.lr = lr
        self.decay = decay
        self.input_shape = input_shape
        self.image_size = self.input_shape[1]
        self.lr_coefficient = lr_coefficient
        self.decay_coefficient = decay_coefficient

        self.generator = Model()
        self.discriminator = Model()
        self.adversarial = Model()

        self.build_models()

    @staticmethod
    def build_generator(inputs, image_size):

        image_resize = image_size // 4
        kernel_size = 5
        layer_filters = [128, 64, 32, 1]

        x = Dense(image_resize * image_resize * layer_filters[0])(inputs)
        x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

        for filters in layer_filters:
            if filters > layer_filters[-2]:
                strides = 2
            else:
                strides = 1
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='same')(x)

        x = Activation('sigmoid')(x)
        generator = Model(inputs, x, name='generator')
        return generator

    @staticmethod
    def build_discriminator(inputs):

        kernel_size = 5
        layer_filters = [32, 64, 128, 256]

        x = inputs
        for filters in layer_filters:
            if filters == layer_filters[-1]:
                strides = 1
            else:
                strides = 2
            x = LeakyReLU(alpha=0.2)(x)
            x = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same')(x)

        x = Flatten()(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)
        discriminator = Model(inputs, x, name='discriminator')
        return discriminator

    def build_models(self):

        discriminator_input_shape = self.input_shape
        discriminator_input = Input(shape=discriminator_input_shape, name='discriminator_input')
        self.discriminator = self.build_discriminator(discriminator_input)

        discriminator_optimizer = RMSprop(lr=self.lr, decay=self.decay)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=discriminator_optimizer,
                                   metrics=['accuracy'])

        generator_input_shape = (self.latent_size,)
        gen_adv_input = Input(shape=generator_input_shape, name='generator_input')
        self.generator = self.build_generator(gen_adv_input, self.image_size)

        adversarial_optimizer = RMSprop(lr=self.lr * self.lr_coefficient, decay=self.decay * self.decay_coefficient)
        self.discriminator.trainable = False
        self.adversarial = Model(gen_adv_input,
                                 self.discriminator(self.generator(gen_adv_input)),
                                 name=self.model_name)
        self.adversarial.compile(loss='binary_crossentropy',
                                 optimizer=adversarial_optimizer,
                                 metrics=['accuracy'])

    def train(self, x_train):

        save_interval = self.save_interval
        train_steps = self.train_steps
        noise_input = np.random.uniform(-1.0, 1.0, size=[16, self.latent_size])
        train_size = x_train.shape[0]

        for i in range(train_steps):
            rand_indexes = np.random.randint(0, train_size, size=self.batch_size)
            real_images = x_train[rand_indexes]
            generator_noise = np.random.uniform(-1.0,
                                                1.0,
                                                size=[self.batch_size, self.latent_size])

            fake_images = self.generator.predict(generator_noise)
            x = np.concatenate((real_images, fake_images))
            y = np.ones([2 * self.batch_size, 1])
            y[self.batch_size:, :] = 0.0
            loss, acc = self.discriminator.train_on_batch(x, y)
            log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

            adversarial_noise = np.random.uniform(-1.0,
                                                  1.0,
                                                  size=[self.batch_size, self.latent_size])

            y = np.ones([self.batch_size, 1])

            loss, acc = self.adversarial.train_on_batch(adversarial_noise, y)
            log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
            print(log)
            if (i + 1) % save_interval == 0:
                self.plot_images(noise_input=noise_input,
                                 show=False,
                                 step=(i + 1),
                                 model_name=self.model_name)

        self.generator.save(self.model_name + ".h5")

    def plot_images(self,
                    noise_input,
                    show=False,
                    step=0,
                    model_name="gan"):

        os.makedirs(model_name, exist_ok=True)
        filename = os.path.join(model_name, "%05d.png" % step)
        images = self.generator.predict(noise_input)
        plt.figure(figsize=(2.2, 2.2))
        num_images = images.shape[0]
        image_size = images.shape[1]
        rows = int(math.sqrt(noise_input.shape[0]))
        for i in range(num_images):
            plt.subplot(rows, rows, i + 1)
            image = np.reshape(images[i], [image_size, image_size])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.savefig(filename)
        if show:
            plt.show()
        else:
            plt.close('all')

    def train_models(self, img_normal=255):

        (x_train, _), (_, _) = mnist.load_data()

        image_size = x_train.shape[1]
        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        x_train = x_train.astype('float32') / img_normal

        self.train(x_train)

    @staticmethod
    def test_generator(generator):

        noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        generator.plot_images(noise_input=noise_input,
                              show=True,
                              model_name="test_outputs")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    args = parser.parse_args()
    dc_gan = DCGAN()

    if args.generator:
        test_generator = load_model(args.generator)
        dc_gan.test_generator(test_generator)
    else:
        dc_gan.train_models()
