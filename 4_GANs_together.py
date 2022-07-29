import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Dense, Reshape, \
    BatchNormalization, Conv2DTranspose, LeakyReLU, Conv2D, Flatten, ReLU

from tensorflow.keras.optimizers import RMSprop

from datetime import datetime
from json import loads
from pathlib import Path
import os
import math

from tensorflow.keras.datasets import mnist


#%%

class AbstractDiscriminator(Model):
    def __init__(self, *args, **kwargs):
        super(AbstractDiscriminator, self).__init__(*args, **kwargs)


class AbstractGenerator(Model):
    def __init__(self, *args, **kwargs):
        super(AbstractGenerator, self).__init__(*args, **kwargs)


#%%

class PrepareTrainData:

    def __init__(self,
                 dataset_name: str = 'mnist',
                 path: os.path = os.path.dirname(__file__),
                 buffer_size: int = 60000,
                 batch_size: int = 32):
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(path, 'data', dataset_name)
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        datasets = {
            'mnist': self._prepare_mnist(),
            'one_dim': self._prepare_distribution()
        }

        self.dataset = datasets[dataset_name]

    def _prepare_mnist(self):
        if os.path.exists(self.dataset_path):
            pass
            # The function is deprecated. Use tf.data.Dataset.load() instead
            # return tf.data.experimental.load(self.dataset_path)

        (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
        x_train = (x_train - 127.5) / 127.5
        # x_train = x_train / 255.0

        y_train = tf.keras.utils.to_categorical(y_train)

        dataset = tf.data.Dataset\
            .from_tensor_slices((x_train, y_train))\
            .shuffle(self.buffer_size)\
            .batch(self.batch_size)

        tf.data.experimental.save(dataset, self.dataset_path)
        # dataset.save(self.dataset_path)  # 2.9.1 ?

        return dataset

    def _prepare_distribution(self):
        if os.path.exists(self.dataset_path):
            pass
            # The function is deprecated. Use tf.data.Dataset.load() instead
            # return tf.data.experimental.load(self.dataset_path)

        init_sampling = np.random.normal(0, 1, size=self.buffer_size).astype('float32')
        x_train = init_sampling[(init_sampling < -3) |
                                ((-2 < init_sampling) & (init_sampling < -1)) |
                                ((0 < init_sampling) & (init_sampling < 1)) |
                                ((2 < init_sampling) & (init_sampling < 3))]

        dataset = tf.data.Dataset\
            .from_tensor_slices(x_train)\
            .shuffle(self.buffer_size)\
            .batch(self.batch_size)

        tf.data.experimental.save(dataset, self.dataset_path)
        # dataset.save(self.dataset_path)
        
        return dataset
    
    
#%%

class SaveResultsCallback(tf.keras.callbacks.Callback):
    
    def __init__(self,
                 saving_period: int = 500,
                 saving_path: os.path = os.path.dirname(__file__)):
        super(SaveResultsCallback, self).__init__()
        self.saving_period = saving_period
        self.images_saving_path = os.path.join(saving_path, 'img')
        self.models_saving_path = os.path.join(saving_path, 'models')

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.saving_period == 0:
            self._save_images(epoch)
            self._save_models(epoch)

    def _save_images(self, epoch: int):
        image_name = os.path.join(self.img_saving_path, '%05d' % (epoch + 1))
        random_latent_vectors = tf.random.normal(shape=(self.model.batch_size, self.model.latent_dim))
        generated_images = self.model.generator.predict(random_latent_vectors)

        if generated_images.ndim == 2:
            plt.scatter(generated_images[:, 0], np.zeros(generated_images.shape[0]))
            plt.xlim([-4, 4])
            plt.savefig(image_name)
            plt.close('all')

        elif generated_images.ndim == 4:
            plt.figure(figsize=(2.2, 2.2))
            rows = int(math.sqrt(random_latent_vectors.shape[0]))
            for i in range(generated_images.shape[0]):
                plt.subplot(rows, rows, i+1)
                plt.imshow(generated_images[i, :, :, 0], cmap='gray')
                plt.axis('off')
            plt.savefig(image_name)
            plt.close('all')

    def _save_models(self, epoch: int):
        discriminator_path = os.path.join(self.models_saving_path, '%05d_discriminator' % (epoch + 1))
        generator_path = os.path.join(self.models_saving_path, '%05d_generator' % (epoch + 1))

        self.model.discriminator.save(discriminator_path, save_format='tf')
        self.model.generator.save(generator_path, save_format='tf')


#%%

def named_logs(model, logs):
    result = {}
    for log in zip(model.metrics_names, logs):
        result[log[0]] = log[1]
    return result


def loss_fn(labels, output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, output)


#%%

class DiscriminatorDCGAN(AbstractDiscriminator):
    def __init__(self,
                 model_name: str = 'discriminator_for_dcgan',
                 kernel_size: int = 5,
                 layer_filters: tuple = (32, 64, 128, 256),
                 layer_strides: tuple = (2, 2, 2, 1),
                 leaky_relu_alpha: float = 0.2,
                 output_activation: str = 'sigmoid'):
        super(DiscriminatorDCGAN, self).__init__(name=model_name)
        self.kernel_size = kernel_size
        self.layer_filters = layer_filters
        self.layer_strides = layer_strides
        self.leaky_relu_alpha = leaky_relu_alpha

        self.discriminator_blocks = {}

        for index, filters in enumerate(self.layer_filters):
            strides = self.layer_strides[index]
            self.discriminator_blocks[f'block_{index}'] = [
                LeakyReLU(alpha=self.leaky_relu_alpha),
                Conv2D(filters=filters,
                       strides=strides,
                       kernel_size=self.kernel_size,
                       padding='same',
                       name=f'conv_{index}'
                       )
            ]

        self.flatten_ = Flatten(name='flatten')
        self.output_ = Dense(1, name='output')
        self.output_activation = Activation(output_activation)

    # @tf.autograph.experimental.do_not_convert
    def call(self,
             x,
             training=False,
             mask=None):
        for index, filters in enumerate(self.layer_filters):
            for block_ in self.discriminator_blocks[f'block_{index}']:
                x = block_(x)
        x = self.flatten_(x)
        x = self.output_(x)
        x = self.output_(x)

        return x


class GithubDiscriminator(AbstractDiscriminator):
    def __init__(self,
                 model_name: str = 'github_discriminator'):
        super(GithubDiscriminator, self).__init__(name=model_name)

        self.convolution_1 = Conv2D(filters=64,
                                    kernel_size=5,
                                    strides=2,
                                    input_shape=(28, 28, 1),
                                    padding='same',
                                    name='convolution_1')

        self.convolution_2 = Conv2D(filters=128,
                                    kernel_size=5,
                                    strides=2,
                                    padding='same',
                                    name='convolution_2')

        self.flatten_ = Flatten(name='flatten')
        self.output_ = Dense(1, name='output')

    def call(self,
             x,
             training=False,
             mask=None):
        x = tf.nn.leaky_relu(self.convolution_1(x))
        if training:
            x = tf.nn.dropout(x, 0.3)

        x = tf.nn.relu(self.convolution_2(x))
        if training:
            x = tf.nn.dropout(x, 0.3)

        x = self.flatten_(x)
        return self.output_(x)


class OneDimDiscriminator(AbstractDiscriminator):
    def __init__(self,
                 model_name: str = 'one_dim_discriminator',
                 layer_filters: tuple = (8, 8, 8),
                 output_dim: int = 1,
                 output_activation: str = 'linear'):
        super(OneDimDiscriminator, self).__init__(name=model_name)
        self.layer_filters = layer_filters
        self.output_dim = output_dim

        self.discriminator_blocks = []

        for filters in self.layer_filters:
            self.discriminator_blocks.append(Dense(filters, activation='relu'))  # LeakyReLU?

        self.output_ = Dense(self.output_dim, activation=output_activation)

    def call(self,
             x,
             training=False,
             mask=None):
        for block_ in self.discriminator_blocks:
            x = block_(x)

        x = self.output_(x)

        return x


class DiscriminatorConditionalGAN(AbstractDiscriminator):
    def __init__(self,
                 model_name: str = 'discriminator_cgan',
                 kernel_size: int = 5,
                 layer_filters: tuple = (32, 64, 128, 256),
                 layer_strides: tuple = (2, 2, 2, 1),
                 leaky_relu_alpha: float = 0.2,
                 output_activation: str = 'sigmoid'
                 ):
        super(DiscriminatorConditionalGAN, self).__init__(name=model_name)
        self.kernel_size = kernel_size
        self.layer_filters = layer_filters
        self.layer_strides = layer_strides
        self.leaky_relu_alpha = leaky_relu_alpha
        self.output_activation = output_activation

        self.discriminator_blocks = {}

        for index, filters in enumerate(self.layer_filters):
            strides = self.layer_strides[index]
            self.discriminator_blocks[f'block_{index}'] = [
                LeakyReLU(alpha=self.leaky_relu_alpha),
                Conv2D(filters=filters,
                       kernel_size=self.kernel_size,
                       strides=strides,
                       padding='same',
                       name=f'conv_{index}')
            ]

        self.flatten_ = Flatten(name='flatten')
        self.output_ = Dense(1, name='output')
        self.output_activation = Activation(output_activation)

    def call(self,
             x,
             training=None,
             mask=None):
        for index, filters in enumerate(self.layer_filters):
            for block_ in self.discriminator_blocks[f"block_{index + 1}"]:
                x = block_(x)

        x = self.flatten_(x)
        x = self.output_(x)
        x = self.output_activation(x)

        return x


#%%

class GeneratorDCGAN(AbstractGenerator):
    def __init__(self,
                 model_name: str = 'generator_dcgan',
                 image_size: int = 28,
                 kernel_size: int = 5,
                 layer_filters: tuple = (128, 64, 32, 1),
                 layer_strides: tuple = (2, 2, 1, 1),
                 leaky_relu_alpha: float = 0.2,
                 output_activation: str = 'tanh',
                 ):
        super(GeneratorDCGAN, self).__init__(name=model_name)
        self.image_resize = image_size // 4
        self.kernel_size = kernel_size
        self.layer_filters = layer_filters
        self.layer_strides = layer_strides
        self.leaky_relu_alpha = leaky_relu_alpha

        self.generator_blocks = {}

        for index, filters in enumerate(self.layer_filters):
            strides = self.layer_strides[index]
            self.generator_blocks[f'block_{index}'] = [
                BatchNormalization(name=f'batch_norm_{index}'),
                LeakyReLU(alpha=self.leaky_relu_alpha),
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
             x,
             training=None,
             mask=None):
        x = self.input_dense(x)
        x = self.input_reshape(x)

        for index, filters in enumerate(self.layer_filters):
            for block_ in self.blocks[f"block_{index}"]:
                x = block_(x)

        x = self.output_activation(x)

        return x


class GithubGenerator(AbstractGenerator):
    def __init__(self,
                 model_name: str = 'github_generator'):
        super(GithubGenerator, self).__init__(name=model_name)
        self.input_dense = Dense(7 * 7 * 256, use_bias=False, input_shape=(100,), name="input_dense")
        self.batch_norm_1 = BatchNormalization(name='batch_norm_1')
        self.reshape_ = Reshape((7, 7, 256), name="reshape")

        self.conv_transpose_1 = Conv2DTranspose(filters=128,
                                                kernel_size=5,
                                                strides=1,
                                                use_bias=False,
                                                padding='same',
                                                name='conv_transpose_1')

        self.batch_norm_2 = BatchNormalization(name='batch_norm_2')

        self.conv_transpose_2 = Conv2DTranspose(filters=64,
                                                kernel_size=5,
                                                strides=2,
                                                use_bias=False,
                                                padding="same",
                                                name="conv_transpose_2")

        self.batch_norm_3 = BatchNormalization(name='batch_norm_3')

        self.conv_transpose_3 = Conv2DTranspose(filters=1,
                                                kernel_size=5,
                                                strides=2,
                                                use_bias=False,
                                                activation='tanh',
                                                padding="same",
                                                name="conv_transpose_2")

    def call(self,
             x,
             training=None,
             mask=None):
        x = self.input_dense(x)
        x = tf.nn.leaky_relu(self.batch_norm_1(x))

        x = self.reshape_(x)

        x = self.conv_transpose_1(x)
        x = tf.nn.leaky_relu(self.batch_norm_2(x))

        x = self.conv_transpose_2(x)
        x = tf.nn.leaky_relu(self.batch_norm_3(x))

        x = self.conv_transpose_3(x)

        return x


class OneDimGenerator(AbstractGenerator):
    def __init__(self,
                 model_name: str = 'one_dim_generator',
                 layer_filters: tuple = (8, 8, 8),
                 output_dim: int = 1,
                 output_activation: str = 'linear'  # softmax?
                 ):
        super(OneDimGenerator, self).__init__(name=model_name)
        self.layer_filters = layer_filters
        self.output_dim = output_dim

        self.generator_blocks = []

        for filters in self.layer_filters:
            self.generator_blocks.append(Dense(filters, activation='relu'))  # LeakyReLU?

        self.output_ = Dense(self.output_dim, activation=output_activation)

    def call(self,
             x,
             training=None,
             mask=None):
        for block_ in self.generator_blocks:
            x = block_(x)

        x = self.output_(x)

        return x


class GeneratorConditionalGAN(AbstractGenerator):
    def __init__(self,
                 model_name: str = 'conditional_generator',
                 image_size: int = 28,
                 kernel_size: int = 5,
                 layer_filters: tuple = (128, 64, 32, 1),
                 layer_strides: tuple = (2, 2, 1, 1),
                 leaky_relu_alpha: float = 0.2,
                 output_activation: str = 'sigmoid',):
        super(GeneratorConditionalGAN, self).__init__(name=model_name)
        self.image_resize = image_size // 4
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.layer_filters = layer_filters
        self.layer_strides = layer_strides
        self.leaky_relu_alpha = leaky_relu_alpha

        self.generator_blocks = {}

        for index, filters in enumerate(self.layer_filters):
            strides = self.layer_strides[index]
            self.generator_blocks[f'block_{index}'] = [
                LeakyReLU(alpha=self.leaky_relu_alpha),
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
             x,
             training=None,
             mask=None):
        x = self.input_dense(x)
        x = self.input_reshape(x)

        for index, filters in enumerate(self.layer_filters):
            for block_ in self.blocks[f"block_{index}"]:
                x = block_(x)

        x = self.output_activation(x)

        return x


#%%

class ConditionalAdversarialNetwork(Model):
    def __init__(self,
                 generator: AbstractGenerator,
                 discriminator: AbstractDiscriminator,
                 latent_dim: int = 100,
                 batch_size: int = 32,  # 32
                 model_name: str = 'adversarial_network',
                 num_classes: int = 10  # for conditional gan
                 ):
        super(ConditionalAdversarialNetwork, self).__init__(name=model_name)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.d_optimizer = None
        self.g_optimizer = None
        self.loss_fn = None

    def compile(self,
                g_optimizer: tf.keras.optimizers = None,
                d_optimizer: tf.keras.optimizers = None,
                loss_function: tf.keras.losses = None
                ):
        super(ConditionalAdversarialNetwork, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_function

    @tf.function
    def train_step(self, data):
        real_images, one_hot_labels = data

        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[self.generator.image_size * self.generator.image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, self.generator.image_size, self.generator.image_size, self.num_classes)
        )

        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        generated_images = self.generator(random_vector_labels)

        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        labels = tf.concat(
            [tf.ones((self.batch_size, 1)), tf.zeros((self.batch_size, 1))], axis=0
        )

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        misleading_labels = tf.zeros((self.batch_size, 1))

        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"g_loss": g_loss, "d_loss": d_loss}


class AdversarialNetwork(Model):

    def __init__(self,
                 generator: AbstractGenerator,
                 discriminator: AbstractDiscriminator,
                 latent_dim: int = 100,
                 batch_size: int = 100,  # 32
                 model_name: str = 'adversarial_network',
                 ):
        super(AdversarialNetwork, self).__init__(name=model_name)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.batch_size = batch_size

        self.d_optimizer = None
        self.g_optimizer = None
        self.loss_fn = None

    def compile(self,
                g_optimizer: tf.keras.optimizers = None,
                d_optimizer: tf.keras.optimizers = None,
                loss_function: tf.keras.losses = None
                ):
        super(AdversarialNetwork, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_function

    @tf.function
    def train_step(self, data):
        real_images = data
        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        generated_images = self.generator(random_latent_vectors)
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat([tf.ones((self.batch_size, 1)),
                            tf.zeros((self.batch_size, 1))], axis=0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        misleading_labels = tf.zeros((self.batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}


#%%

