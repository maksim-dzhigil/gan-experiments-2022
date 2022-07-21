from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
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


class AdversarialNetwork(tf.keras.Model):

    def __init__(self,
                 name='adv',
                 generated_object_dim=1,
                 ):
        super(AdversarialNetwork, self).__init__(name=name)
        self.generated_object_dim = generated_object_dim

        self.model_generator = GeneratorDense(name=name+"_gen", last_layer_shape=generated_object_dim)

        self.model_discriminator = DiscriminatorDense(name=name+"_dis")
        self.model_discriminator.trainable = False  # WHY!?!?!?!?!

    def call(self, inputs):

        gen_results = self.model_generator(inputs)
        dis_assessments = self.model_discriminator(gen_results)

        return dis_assessments


class SimpleDenseModel(tf.keras.Model):

    def __init__(self,
                 name,
                 layer_size_list=[8, 8, 8],
                 last_layer_shape=1,
                 last_layer_activation='linear'
                 ):
        super(SimpleDenseModel, self).__init__(name=name)
        self.layer_size_list = layer_size_list
        self.last_layer_shape = last_layer_shape

        self.layer_list = []
        for layer_size in layer_size_list:
            self.layer_list.append(Dense(layer_size, activation='relu'))

        self.the_last_layer = Dense(self.last_layer_shape, activation=last_layer_activation)

    def call(self, inputs, training=True):

        x = inputs
        for layer_ in self.layer_list:
            x = layer_(x)

        x = self.the_last_layer(x)

        return x


class GeneratorDense(SimpleDenseModel):

    def __init__(self, name='simple_dense_generator',
                 layer_size_list=[8, 8, 8],
                 last_layer_shape=1):
        super(GeneratorDense, self).__init__(name=name,
                                             layer_size_list=layer_size_list,
                                             last_layer_shape=last_layer_shape,
                                             last_layer_activation = 'linear')


class DiscriminatorDense(SimpleDenseModel):

    def __init__(self, name='simple_dense_discriminator',
                 layer_size_list = [8, 8, 8]):
        super(DiscriminatorDense, self).__init__(name=name,
                                             layer_size_list=layer_size_list,
                                             last_layer_shape=1,
                                             last_layer_activation = 'sigmoid')

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=100,
                         train_steps=10000,
                         latent_size = 1,
                         gen_vec_size=1,
                         ):

        self.batch_size = batch_size
        self.train_steps = train_steps
        self.latent_size = latent_size
        self.gen_vec_size = gen_vec_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return 1

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        X_noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.latent_size])

        init_sampling = np.random.normal(0,1, size=100000)
        Y_target_sampling = init_sampling[((init_sampling < -3)) |
                                            ((-2 < init_sampling) & (init_sampling < -1)) |
                                            ((0 < init_sampling) & (init_sampling < 1)) |
                                            ((2 < init_sampling) & (init_sampling < 3))]

        Y_target_sampling = np.random.choice(Y_target_sampling, 100)[:,np.newaxis]

        return X_noise, Y_target_sampling



class DCGANTraining:

    def __init__(self,
                 name='DCGAN',
                 ):

        self.discriminator = None
        self.generator = None
        self.adversarial = None


    def build_models(self,
                     latent_size=(1,),
                     gen_vec_size=(1,),
                     lr=2e-4,
                     decay=6e-8,
                     lr_multiplier=0.5,
                     decay_multiplier=0.5):
        pass # TODO

    def train_models(self,
                     batch_size=100,
                     train_steps=10000,
                     log_folder='./',
                     ):
        pass # TODO



    