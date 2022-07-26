from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Activation, Dense, Input, Reshape, BatchNormalization, Conv2DTranspose, LeakyReLU, Conv2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import math
import datetime
import os

from inspect import signature

from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Flatten


class ImageGeneratorBasic(tf.keras.Model):

    def __init__(self,
                 name='generator',
                 image_size=28,
                 kernel_size=5,
                 layer_filters=(128, 64, 32, 1)
                 ):

        super(ImageGeneratorBasic, self).__init__(name=name)
        self.image_resize = image_size // 4
        self.kernel_size = kernel_size
        self.layer_filters = layer_filters

        self.layer_010_dense = Dense(self.image_resize * self.image_resize * layer_filters[0])
        self.layer_020_reshape = Reshape((self.image_resize, self.image_resize, layer_filters[0]))

        self.block_dict = {}

        for ind, filters in enumerate(self.layer_filters):
            if filters > layer_filters[-2]:
                strides = 2
            else:
                strides = 1

            self.block_dict[f"block_{ind}"] = [
                BatchNormalization(),
                Activation('relu'),
                Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='same')
            ]

        self.layer_050_activation = Activation('sigmoid')

    def call(self,
             inputs,
             training=True,
             mask=None):

        x = self.layer_010_dense(inputs)
        x = self.layer_020_reshape(x)

        for ind, filters in enumerate(self.layer_filters):
            for layer_ in self.block_dict[f"block_{ind}"]:
                x = layer_(x)

        x = self.layer_050_activation(x)
        print(x.shape)
        return x


class ImageDiscriminatorBasic(tf.keras.Model):

    def __init__(self,
                 name='discriminator',
                 kernel_size=5,
                 layer_filters=(32, 64, 128, 256)
                 ):

        super(ImageDiscriminatorBasic, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.layer_filters = layer_filters

        self.block_dict = {}
        for ind, filters in enumerate(self.layer_filters):
            if filters == layer_filters[-1]:
                strides = 1
            else:
                strides = 2
            self.block_dict[f"block_{ind}"] = [
                LeakyReLU(alpha=0.2),
                Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same')
            ]

        self.layer_head_010 = Flatten()
        self.layer_head_020 = Dense(1)
        self.layer_head_030 = Activation('sigmoid')

    def call(self,
             inputs,
             training=True,
             mask=None):

        x = inputs
        print(x.shape)
        for ind, filters in enumerate(self.layer_filters):
            for layer_ in self.block_dict[f"block_{ind}"]:
                x = layer_(x)

        x = self.layer_head_010(x)
        x = self.layer_head_020(x)
        x = self.layer_head_030(x)
        return x


class AdversarialNetwork(tf.keras.Model):

    def __init__(self,
                 generator: tf.keras.Model,
                 discriminator: tf.keras.Model,
                 name='adv'
                 ):
        super(AdversarialNetwork, self).__init__(name=name)

        self.model_generator = generator
        self.model_discriminator = discriminator

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


class DataGeneratorForImageGAN(tf.keras.utils.Sequence):

    def __init__(self,
                 batch_size=100,
                 latent_space_size=1,
                 noise_only=False
                 ):

        self.batch_size = batch_size
        self.latent_space_size = latent_space_size
        self.noise_only = noise_only

        (x_train, _), (_, _) = mnist.load_data()

        # reshape data for CNN as (28, 28, 1) and normalize
        image_size = x_train.shape[1]
        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        self.x_train = x_train.astype('float32') / 255

    def __len__(self):
        return 1

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):

        if self.noise_only:
            X_noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.latent_space_size])

            return X_noise
        else:
            X_noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.latent_space_size])

            rand_indexes = np.random.randint(0, self.x_train.shape[0], size=self.batch_size)

            return X_noise, self.x_train[rand_indexes]


class DataGeneratorForDenseGAN(tf.keras.utils.Sequence):

    def __init__(self,
                 batch_size=100,
                 latent_space_size=1,
                 noise_only=False
                 ):

        self.batch_size = batch_size
        self.latent_space_size = latent_space_size
        self.noise_only = noise_only

    def __len__(self):
        return 1

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):

        if self.noise_only:
            X_noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.latent_space_size])

            return X_noise
        else:
            X_noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.latent_space_size])

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
                 latent_size=(None, 1),
                 gen_vec_size=(None, 1),
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

        self.discriminator = None
        self.generator = None
        self.adversarial = None
        self.sample_generator = DataGeneratorForDenseGAN()

        self.dis_callback = None
        self.adv_callback = None

    def build_models(self,
                     generator,
                     discriminator,
                     lr=2e-4,
                     decay=6e-8,
                     lr_multiplier=0.5,
                     decay_multiplier=0.5,):

        self.generator = generator  # GeneratorDense(name=self.name+"_gen", output_shape_=self.gen_vec_size)
        self.generator.build(input_shape=self.latent_size)

        self.discriminator = discriminator  # DiscriminatorDense(name=self.name+"_dis")
        self.discriminator.build(input_shape=self.gen_vec_size)
        dis_optimizer = RMSprop(learning_rate=lr, decay=decay)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=dis_optimizer,
                                   metrics=['accuracy'])
        self.discriminator.trainable = False  # WHY!?!?!?!?!

        self.adversarial = AdversarialNetwork(self.generator, self.discriminator)
        self.adversarial.build(input_shape=self.latent_size)
        adv_optimizer = RMSprop(learning_rate=lr*lr_multiplier, decay=decay*decay_multiplier)
        self.adversarial.compile(loss='binary_crossentropy',
                                 optimizer=adv_optimizer,
                                 metrics=['accuracy'])

        self._create_callback()

    def train_models(self,
                     train_datagen_gen : tf.keras.utils.Sequence,
                     train_datagen_adv: tf.keras.utils.Sequence,
                     test_gen: tf.keras.utils.Sequence,
                     train_steps=10000,
                     save_interval=500,
                     ):
        # preparing training data
        x_train_noise, y_train_sample = next(iter(train_datagen_gen))
        train_datagen_adv.noise_only = True
        x_noise_for_adv = next(iter(train_datagen_adv))
        x_test_noise, _ = next(iter(test_gen))
        # training process
        for i in range(train_steps):
            real_distribution = y_train_sample
            fake_distribution = self.generator.predict(x_train_noise)
            # preparing training data for discriminator
            x_train_dis = np.concatenate((real_distribution, fake_distribution))
            y_train_dis = np.ones([2 * real_distribution.shape[0], 1])
            y_train_dis[real_distribution.shape[0]:, :] = 0.0
            # preparing training data for adversarial
            x_train_adv = x_noise_for_adv
            y_train_adv = np.ones([x_train_adv.shape[0], 1])
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
            histogram_freq=0
        )
        self.adv_callback = TensorBoard(
            log_dir=self.folder_adv_log,
            histogram_freq=0
        )

        self.dis_callback.set_model(self.discriminator)
        self.adv_callback.set_model(self.adversarial)

    def _save_results(self,
                      x_noise,
                      step):

        img_name = os.path.join(self.folder_img, "%05d.png" % step)
        gen_predict = self.generator.predict(x_noise)

        if gen_predict.ndim == 2:
            plt.scatter(gen_predict[:, 0], np.zeros(gen_predict.shape[0]), marker='|')
            plt.xlim([-4, 4])
            plt.savefig(img_name)
            plt.close('all')

        elif gen_predict.ndim == 4:

            plt.figure(figsize=(2.2, 2.2))
            rows = int(math.sqrt(x_noise.shape[0]))
            for i in range(gen_predict.shape[0]):
                plt.subplot(rows, rows, i + 1)
                plt.imshow(gen_predict[i, :, :, 0], cmap='gray')
                plt.axis('off')
            plt.savefig(img_name)
            plt.close('all')


        # TODO save model structure here
        self.generator.save(os.path.join(self.folder_gen_weights, "%05d" % step), save_format='tf')
        self.discriminator.save(os.path.join(self.folder_dis_weights, "%05d" % step), save_format='tf')


def named_logs(model, logs):
    result = {}
    for log in zip(model.metrics_names, logs):
        result[log[0]] = log[1]
    return result


if __name__ == '__main__':
    EXPERIMENT = 'basic'

    if EXPERIMENT == 'basic':

        name = EXPERIMENT
        dcgan = DCGANTraining(name=name,
                              latent_size=(None, 100),
                              gen_vec_size=(None, 28, 28, 1))

        dis = ImageDiscriminatorBasic(name=name + "_dis")
        gen = ImageGeneratorBasic(name=name + "_gen")
        dcgan.build_models(gen, dis)

        train_datagen_gen = DataGeneratorForImageGAN(batch_size=100, latent_space_size=100)
        train_datagen_adv = DataGeneratorForImageGAN(batch_size=100, latent_space_size=100, noise_only=True)
        test_gen = DataGeneratorForDenseGAN(batch_size=16, latent_space_size=100)
        dcgan.train_models(train_datagen_gen, train_datagen_adv, test_gen)

    elif EXPERIMENT == 'dense':
        name = EXPERIMENT
        dcgan = DCGANTraining(name=name, gen_vec_size=(None,1))

        generator = GeneratorDense(name=name + "_gen", output_shape_=dcgan.gen_vec_size[1])
        discriminator = DiscriminatorDense(name=name + "_dis", )
        dcgan.build_models(generator, discriminator)

        train_datagen_gen = DataGeneratorForDenseGAN()
        train_datagen_adv = DataGeneratorForDenseGAN(noise_only=True)
        test_gen = DataGeneratorForDenseGAN(batch_size=100, latent_space_size=1)
        dcgan.train_models(train_datagen_gen, train_datagen_adv, test_gen)
