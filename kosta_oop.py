from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import math
import datetime
import os

from inspect import signature



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

    def call(self,
             inputs,
             training=None,
             mask=None):

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

    def call(self,
             inputs,
             training=True,
             mask=None):

        x = inputs
        for layer_ in self.layer_list:
            x = layer_(x)

        x = self.the_last_layer(x)

        return x


class GeneratorDense(SimpleDenseModel):

    def __init__(self,
                 name='simple_dense_generator',
                 layer_size_list=[8, 8, 8],
                 last_layer_shape=1):
        super(GeneratorDense, self).__init__(name=name,
                                             layer_size_list=layer_size_list,
                                             last_layer_shape=last_layer_shape,
                                             last_layer_activation='linear')


class DiscriminatorDense(SimpleDenseModel):

    def __init__(self,
                 name='simple_dense_discriminator',
                 layer_size_list=[8, 8, 8]):
        super(DiscriminatorDense, self).__init__(name=name,
                                                 layer_size_list=layer_size_list,
                                                 last_layer_shape=1,
                                                 last_layer_activation='sigmoid')


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 batch_size=100,
                 latent_space_dim=1,
                 ):

        self.batch_size = batch_size
        self.latent_space_dim = latent_space_dim

    def __len__(self):
        return 1

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        X_noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.latent_space_dim])

        init_sampling = np.random.normal(0, 1, size=100000)
        Y_target_sampling = init_sampling[(init_sampling < -3) |
                                          ((-2 < init_sampling) & (init_sampling < -1)) |
                                          ((0 < init_sampling) & (init_sampling < 1)) |
                                          ((2 < init_sampling) & (init_sampling < 3))]

        Y_target_sampling = np.random.choice(Y_target_sampling, 100)[:, np.newaxis]

        return X_noise, Y_target_sampling


class DCGANTraining:

    def __init__(self,
                 name='one_dim_DCGAN',
                 latent_size=1,
                 gen_vec_size=1,
                 lr=2e-4,
                 decay=6e-8,
                 lr_multiplier=0.5,
                 decay_multiplier=0.5,
                 train_steps=10000,
                 batch_size=100,
                 save_interval=1000
                 ):

        self.name = name
        self.directory = f'./{self.name}/'
        self.latent_size = latent_size
        self.gen_vec_size = gen_vec_size
        self.lr = lr
        self.decay = decay
        self.lr_multiplier = lr_multiplier
        self.decay_multiplier = decay_multiplier
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.save_interval = save_interval

        self.discriminator = None
        self.generator = None
        self.adversarial = None
        self.sample_generator = DataGenerator()

        self.dis_callback = None
        self.adv_callback = None

        self.build_models()  # is it worth?
        self.create_callback()

    def build_models(self):
        # instances creation
        self.discriminator = DiscriminatorDense()
        self.generator = GeneratorDense()
        self.adversarial = AdversarialNetwork()
        # building models
        self.discriminator.build(input_shape=(None, self.gen_vec_size))
        self.generator.build(input_shape=(None, self.latent_size))
        self.adversarial.build(input_shape=(None, self.latent_size))
        # preparing optimizers
        dis_optimizer = RMSprop(learning_rate=self.lr, decay=self.decay)
        adv_optimizer = RMSprop(learning_rate=self.lr*self.lr_multiplier, decay=self.decay*self.decay_multiplier)
        # models compiling
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=dis_optimizer,
                                   metrics=['accuracy'])
        self.adversarial.compile(loss='binary_crossentropy',
                                 optimizer=adv_optimizer,
                                 metrics=['accuracy'])

    def train_models(self):
        # preparing training data
        sample_generator = DataGenerator()
        x_train_noise, y_target_sample = next(iter(sample_generator))  # x_train_noise.shape == y_target_sample.shape == (100,1)
        x_test_noise = np.random.uniform(-1.0, 1.0, size=[100, self.latent_size])
        train_size = y_target_sample.shape[0]
        # training process
        for i in range(self.train_steps):
            # making real and fake distributions
            rand_indexes = np.random.randint(0, train_size, size=self.batch_size)
            real_distribution = np.array(y_target_sample[rand_indexes])  # 100 random indexes in 100-elements array. WTF
            fake_distribution = self.generator.predict(x_train_noise)
            # preparing training data for discriminator
            x_train_dis = np.concatenate((real_distribution, fake_distribution))
            y_train_dis = np.ones([2 * self.batch_size, 1])
            y_train_dis[self.batch_size:, :] = 0.0
            # preparing training data for adversarial
            x_train_adv = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.latent_size])
            y_train_adv = np.ones([self.batch_size, 1])
            # computing loss and accuracy
            dis_loss, dis_accuracy = self.discriminator.train_on_batch(x_train_dis, y_train_dis)
            adv_loss, adv_acc = self.adversarial.train_on_batch(x_train_adv, y_train_adv)
            # saving training history for TensorBoard chart
            self.dis_callback.on_epoch_end(i, named_logs(self.discriminator, (dis_loss, dis_accuracy)))
            self.adv_callback.on_epoch_end(i, named_logs(self.adversarial, (adv_loss, adv_acc)))
            # output the results to the console
            log = "%d: [discriminator loss: %f, acc: %f] [adversarial loss: %f, acc: %f]" % (i, dis_loss, dis_accuracy,
                                                                                             adv_loss, adv_acc)
            print(log)

            if (i + 1) % self.save_interval == 0:
                self.save_results(x_test_noise, step=i+1)

            self.dis_callback.on_epoch_end(None)
            self.adv_callback.on_epoch_end(None)

    def create_callback(self):

        self.dis_callback = TensorBoard(
            log_dir=self.directory + 'logs/dis_callback',
            histogram_freq=0,
            batch_size=self.batch_size
        )
        self.adv_callback = TensorBoard(
            log_dir=self.directory + 'logs/adv_callback',
            histogram_freq=0,
            batch_size=self.batch_size
        )

        self.dis_callback.set_model(self.discriminator)
        self.adv_callback.set_model(self.adversarial)

    def save_results(self,
                     x_noise,
                     step=0):
        os.makedirs(f'{self.directory}img/' + self.name, exist_ok=True)
        img_name = os.path.join(f'{self.directory}img/' + self.name, "%05d.png" % step)
        gen_predict = self.generator.predict(x_noise)
        plt.scatter(gen_predict[:, 0], np.zeros(gen_predict.shape[0]), marker='|')
        plt.xlim([-4, 4])
        plt.savefig(img_name)
        plt.close('all')

        self.generator.save(f'{self.directory + self.name}.h5')


def named_logs(model, logs):
    result = {}
    for log in zip(model.metrics_names, logs):
        result[log[0]] = log[1]
    return result


if __name__ == '__main__':
    dcgan = DCGANTraining()
    dcgan.train_models()
