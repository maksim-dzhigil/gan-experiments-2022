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
                 generator : tf.keras.Model,
                 discriminator : tf.keras.Model,
                 name='adv'
                 ):
        super(AdversarialNetwork, self).__init__(name=name)

        self.model_generator = generator

        self.model_discriminator = discriminator
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
                 output_shape_=1,
                 last_layer_activation='linear'
                 ):
        super(SimpleDenseModel, self).__init__(name=name)
        self.layer_size_list = layer_size_list
        self.output_shape_ = output_shape_

        self.layer_list = []
        for layer_size in layer_size_list:
            self.layer_list.append(Dense(layer_size, activation='relu'))

        self.the_last_layer = Dense(self.output_shape_, activation=last_layer_activation)

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
                 output_shape_=1):
        super(GeneratorDense, self).__init__(name=name,
                                             layer_size_list=layer_size_list,
                                             output_shape_=output_shape_,
                                             last_layer_activation='linear')


class DiscriminatorDense(SimpleDenseModel):

    def __init__(self,
                 name='simple_dense_discriminator',
                 layer_size_list=[8, 8, 8]):
        super(DiscriminatorDense, self).__init__(name=name,
                                                 layer_size_list=layer_size_list,
                                                 output_shape_=1,
                                                 last_layer_activation='sigmoid')


class DataGeneratorForGAN(tf.keras.utils.Sequence):

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
                 path_for_output='./',
                 name='one_dim_DCGAN',
                 latent_size=1,
                 gen_vec_size=1,
                 batch_size=100,
                 ):

        self.name = name
        self.directory = os.path.join(path_for_output, name)
        self.folder_adv_log = os.path.join(self.directory, 'logs/adv_callback')
        self.folder_dis_log = os.path.join(self.directory, 'logs/dis_callback')
        self.folder_img = os.path.join(self.directory, 'img/')
        self.folder_gen_weights = os.path.join(self.directory, 'gen_weights/')
        self.folder_dis_weights = os.path.join(self.directory, 'dis_weights/')
        os.makedirs(self.folder_adv_log, exist_ok=True)
        os.makedirs(self.folder_dis_log, exist_ok=True)
        os.makedirs(self.folder_img, exist_ok=True)
        os.makedirs(self.folder_gen_weights, exist_ok=True)
        os.makedirs(self.folder_dis_weights, exist_ok=True)


        self.latent_size = latent_size
        self.gen_vec_size = gen_vec_size
        self.batch_size = batch_size

        self.discriminator = None
        self.generator = None
        self.adversarial = None
        self.sample_generator = DataGeneratorForGAN()

        self.dis_callback = None
        self.adv_callback = None

    def build_models(self,
                     lr=2e-4,
                     decay=6e-8,
                     lr_multiplier=0.5,
                     decay_multiplier=0.5,):
        # instances creation
        self.discriminator = DiscriminatorDense(name=self.name+"_dis")
        self.generator = GeneratorDense(name=self.name+"_gen", output_shape_=self.gen_vec_size)
        self.adversarial = AdversarialNetwork(self.generator, self.discriminator)
        # building models
        self.discriminator.build(input_shape=(None, self.gen_vec_size))
        self.generator.build(input_shape=(None, self.latent_size))
        self.adversarial.build(input_shape=(None, self.latent_size))
        # preparing optimizers
        dis_optimizer = RMSprop(learning_rate=lr, decay=decay)
        adv_optimizer = RMSprop(learning_rate=lr*lr_multiplier, decay=decay*decay_multiplier)
        # models compiling
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=dis_optimizer,
                                   metrics=['accuracy'])
        self.adversarial.compile(loss='binary_crossentropy',
                                 optimizer=adv_optimizer,
                                 metrics=['accuracy'])

        self._create_callback()

    def train_models(self,
                     train_gen : tf.keras.utils.Sequence,
                     test_gen : tf.keras.utils.Sequence,
                     train_steps = 10000,
                     save_interval = 500,
                     ):
        # preparing training data
        x_train_noise, y_train_sample = next(iter(train_gen))  # x_train_noise.shape == y_target_sample.shape == (100,1)
        x_test_noise, _ = next(iter(test_gen))
        train_size = y_train_sample.shape[0]
        # training process
        for i in range(train_steps):
            # making real and fake distributions
            rand_indexes = np.random.randint(0, train_size, size=self.batch_size)
            real_distribution = np.array(y_train_sample[rand_indexes])  # 100 random indexes in 100-elements array. WTF
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

            if (i + 1) % save_interval == 0:
                self._save_results(x_test_noise, step=i+1)

            self.dis_callback.on_epoch_end(None)
            self.adv_callback.on_epoch_end(None)

    def _create_callback(self):

        self.dis_callback = TensorBoard(
            log_dir=self.folder_dis_log,
            histogram_freq=0,
            batch_size=self.batch_size
        )
        self.adv_callback = TensorBoard(
            log_dir=self.folder_adv_log,
            histogram_freq=0,
            batch_size=self.batch_size
        )

        self.dis_callback.set_model(self.discriminator)
        self.adv_callback.set_model(self.adversarial)

    def _save_results(self,
                     x_noise,
                     step):

        img_name = os.path.join(self.folder_img, "%05d.png" % step)
        gen_predict = self.generator.predict(x_noise)
        plt.scatter(gen_predict[:, 0], np.zeros(gen_predict.shape[0]), marker='|')
        plt.xlim([-4, 4])
        plt.savefig(img_name)
        plt.close('all')

        # TODO save model structure here
        self.generator.save(os.join(self.folder_gen_weights, "%05d.h5" % step))
        self.discriminator.save(os.join(self.folder_dis_weights, "%05d.h5" % step))


def named_logs(model, logs):
    result = {}
    for log in zip(model.metrics_names, logs):
        result[log[0]] = log[1]
    return result


if __name__ == '__main__':

    dcgan = DCGANTraining()
    dcgan.build_models()

    train_gen = DataGeneratorForGAN()
    test_gen = DataGeneratorForGAN()
    dcgan.train_models(train_gen, test_gen)
