from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Dense, Reshape, \
    BatchNormalization, Conv2DTranspose, LeakyReLU, Conv2D, Flatten

from tensorflow.keras.optimizers import RMSprop

from datetime import datetime
from json import loads
from pathlib import Path
import os
import math

from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import Input


class SaveResultsCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 saving_period: int = 500,
                 saving_path: Path = '' #Path(__file__).parent.absolute()
                 ):
        super(SaveResultsCallback, self).__init__()
        self.saving_period = saving_period
        self.img_saving_path = Path.joinpath(saving_path, 'img/')
        self.models_saving_path = Path.joinpath(saving_path, 'models/')

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.saving_period == 0:
            self._saving_images(epoch)
            self._saving_models(epoch)

    def _saving_images(self,
                       epoch: int):
        img_name = Path.joinpath(self.img_saving_path, "%05d.png" % (epoch + 1))
        noise = tf.random.normal(shape=(self.model.batch_size, self.model.latent_dim))
        gen_predict = self.model.generator.predict(noise)
        if gen_predict.ndim == 2:
            plt.scatter(gen_predict[:, 0], np.zeros(gen_predict.shape[0]))
            plt.xlim([-4, 4])
            plt.savefig(img_name)
            plt.close('all')

        elif gen_predict.ndim == 4:
            plt.figure(figsize=(2.2, 2.2))
            rows = int(math.sqrt(noise.shape[0]))
            for i in range(gen_predict.shape[0]):
                plt.subplot(rows, rows, i+1)
                plt.imshow(gen_predict[i, :, :, 0], cmap='gray')
                plt.axis('off')
            plt.savefig(img_name)
            plt.close('all')

    def _saving_models(self,
                       epoch: int):
        dis_name = Path.joinpath(self.models_saving_path, "%05d_discriminator" % (epoch + 1))
        gen_name = Path.joinpath(self.models_saving_path, "%05d_generator" % (epoch + 1))

        self.model.discriminator.save(dis_name, save_format='tf')
        self.model.generator.save(gen_name, save_format='tf')


class AbstractDiscriminator(Model):

    def __init__(self, *kargs, **kvargs):
        super(AbstractDiscriminator, self).__init__(*kargs, **kvargs)


class AbstractGenerator(Model):

    def __init__(self, *kargs, **kvargs):
        super(AbstractGenerator, self).__init__(*kargs, **kvargs)

# TODO add inherit
class AbstractConditionalFeatures():
    def __init__(self, *kargs, **kvargs):
        super(AbstractConditionalFeatures, self).__init__(*kargs, **kvargs)

# TODO add inherit
class AbstractConditionalLabels():
    def __init__(self, *kargs, **kvargs):
        super(AbstractConditionalLabels, self).__init__(*kargs, **kvargs)


class ConditionalDiscriminatorFromBook(AbstractDiscriminator):

    def __init__(self,
                 model_name: str = 'conditional_gan',
                 image_size: int = 28,
                 kernel_size: int = 5,
                 layer_filters: tuple = (32, 64, 128, 256),
                 layer_strides: tuple = (2, 2, 2, 1),
                 leaky_relu_ratio: float = 0.2,
                 output_activation: str = 'sigmoid'
                 ):
        super(ConditionalDiscriminatorFromBook, self).__init__(name=model_name)

        self.image_size = image_size
        self.kernel_size = kernel_size
        self.layer_filters = layer_filters
        self.layer_strides = layer_strides
        self.leaky_relu_ratio = leaky_relu_ratio

        self.blocks = {}

        for index, filters in enumerate(self.layer_filters):
            strides = self.layer_strides[index]
            self.blocks[f'block_{index + 1}'] = [
                LeakyReLU(alpha=self.leaky_relu_ratio),
                Conv2D(filters=filters,
                       kernel_size=self.kernel_size,
                       strides=strides,
                       padding='same',
                       name=f'conv_{index + 1}')
            ]

        self.flatten = Flatten(name='flatten')
        self.output_ = Dense(1, name='output')
        self.output_activation = Activation(output_activation)

        self.label_dense = Dense(self.image_size * self.image_size)
        self.label_reshape = Reshape((self.image_size, self.image_size, 1))

    def call(self,
             x,
             training=False,
             mask=None):


        for index, filters in enumerate(self.layer_filters):
            for layer_ in self.blocks[f"block_{index + 1}"]:
                x = layer_(x)

        x = self.flatten(x)
        x = self.output_(x)
        x = self.output_activation(x)

        return x


class ConditionalGeneratorFromBook(AbstractGenerator):

    def __init__(self,
                 model_name: str = 'conditional_generator',
                 image_size: int = 28,
                 kernel_size: int = 5,
                 layer_filters: tuple = (128, 64, 32, 1),
                 layer_strides: tuple = (2, 2, 1, 1),
                 leaky_relu_ratio: float = 0.2,
                 output_activation: str = 'sigmoid',):
        super(ConditionalGeneratorFromBook, self).__init__(name=model_name)

        self.image_resize = image_size // 4
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.layer_filters = layer_filters
        self.layer_strides = layer_strides
        self.leaky_relu_ratio = leaky_relu_ratio

        self.blocks = {}

        for index, filters in enumerate(self.layer_filters):
            strides = self.layer_strides[index]
            self.blocks[f'block_{index}'] = [
                LeakyReLU(alpha=self.leaky_relu_ratio),
                BatchNormalization(name=f'batch_norm_{index}'),
                Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='same',
                                name=f'conv_{index}')
            ]

        self.input_dense = Dense(self.image_resize * self.image_resize * layer_filters[0])
        self.input_reshape = Reshape((self.image_resize, self.image_resize, layer_filters[0]))
        self.output_activation = Activation(output_activation)

    def call(self,
             inputs,
             training=False,
             mask=None):

        x = self.input_dense(inputs)
        x = self.input_reshape(x)

        for index, filters in enumerate(self.layer_filters):
            for layer_ in self.blocks[f"block_{index}"]:
                x = layer_(x)

        x = self.output_activation(x)

        return x


class AdversarialNetwork(Model):

    def __init__(self,
                 generator: AbstractGenerator,
                 discriminator: AbstractDiscriminator,
                 latent_dim: int,
                 model_name: str = 'adversarial_network',
                 batch_size: int = 32,  # 32
                 num_classes: int = 10  # for conditional gan
                 ):
        super(AdversarialNetwork, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.d_optimizer = None
        self.g_optimizer = None
        self.loss_fn = None

    def compile(self,
                g_optimizer: tf.keras.optimizers = None,
                d_optimizer: tf.keras.optimizers = None,
                loss_fn: tf.keras.losses = None
                ):
        super(AdversarialNetwork, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, data):

        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[self.generator.image_size * self.generator.image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, self.generator.image_size, self.generator.image_size, self.num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((self.batch_size, 1)), tf.zeros((self.batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((self.batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {
            "g_loss": g_loss,
            "d_loss": d_loss,
        }


class PrepareTrainData:

    def __init__(self,
                 dataset_name: str = 'mnist',  # one_dim
                 path: Path = '', #Path(__file__).parent.absolute(),
                 buffer_size: int = 60000,  # 10000
                 batch_size: int = 32,
                 ):

        self.dataset_path = os.path.join(path, 'data', dataset_name)
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        if dataset_name == 'mnist':
            self.dataset = self._prepare_mnist()

    def _prepare_mnist(self):

        (x_train_img, y_train,), (_, _) = tf.keras.datasets.mnist.load_data()
        x_train_img = x_train_img.reshape(x_train_img.shape[0], 28, 28, 1).astype('float32')
        x_train_img = (x_train_img - 127.5) / 127.5
        y_train = keras.utils.to_categorical(y_train.astype('float32'), 10)

        dataset = tf.data.Dataset.from_tensor_slices((x_train_img, y_train)).shuffle(self.buffer_size)\
            .batch(self.batch_size)
        tf.data.experimental.save(dataset, self.dataset_path)

        return dataset


ptd = PrepareTrainData()
dataset = ptd.dataset
gen = ConditionalGeneratorFromBook()
dis = ConditionalDiscriminatorFromBook()

cond_gan = AdversarialNetwork(
    discriminator=dis, generator=gen, latent_dim=100
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

cond_gan.fit(dataset, epochs=20)
