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


# %%

class AbstractDiscriminator(Model):

    def __init__(self, *kargs, **kvargs):
        super(AbstractDiscriminator, self).__init__(*kargs, **kvargs)


class AbstractGenerator(Model):

    def __init__(self, *kargs, **kvargs):
        super(AbstractGenerator, self).__init__(*kargs, **kvargs)


# %%

class ImageDiscriminatorFromBook(AbstractDiscriminator):

    def __init__(self,
                 name: str = 'book_discriminator',
                 kernel_size: int = 5,
                 layer_filters: tuple = (32, 64, 128, 256),
                 layer_strides: tuple = (2, 2, 2, 1),
                 leaky_relu_ratio: float = 0.2,
                 output_activation: str = 'sigmoid',
                 dropout_value: float = None
                 ):
        super(ImageDiscriminatorFromBook, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.layer_filters = layer_filters
        self.layer_strides = layer_strides
        self.leaky_relu_ratio = leaky_relu_ratio
        self.dropout_value = dropout_value

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

    @tf.autograph.experimental.do_not_convert
    def call(self,
             x,
             training=False,
             mask=None):

        for index, filters in enumerate(self.layer_filters):
            for layer_ in self.blocks[f"block_{index + 1}"]:
                x = layer_(x)

        x = self.flatten(x)
        x = self.output_(x)
        x = self.output_activation(x)  # don't use?

        return x


# %%

class ImageGeneratorFromBook(AbstractGenerator):

    def __init__(self,
                 name: str = 'book_generator',
                 image_size: int = 28,
                 kernel_size: int = 5,
                 layer_filters: tuple = (128, 64, 32, 1),
                 layer_strides: tuple = (1, 1, 2, 2),
                 leaky_relu_ratio: float = 0.2,
                 output_activation: str = 'sigmoid',
                 ):
        super(ImageGeneratorFromBook, self).__init__(name=name)

        self.image_resize = image_size // 4
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

    @tf.autograph.experimental.do_not_convert
    def call(self,
             x,
             training=False,
             mask=None):

        x = self.input_dense(x)
        x = self.input_reshape(x)

        for index, filters in enumerate(self.layer_filters):
            for layer_ in self.blocks[f"block_{index}"]:
                x = layer_(x)

        x = self.output_activation(x)

        return x


# %%

class DiscriminatorFromGithub(AbstractDiscriminator):
    def __init__(self,
                 name: str = 'github_discriminator'):
        super(DiscriminatorFromGithub, self).__init__(name=name)
        self.conv1 = keras.layers.Conv2D(64, (5, 5), strides=(2, 2),
                                         padding="same", input_shape=(28, 28, 1),
                                         name="Conv1"
                                         )

        self.conv2 = keras.layers.Conv2D(128, (5, 5), strides=(2, 2),
                                         padding="same",
                                         name="Conv2"
                                         )
        self.flatten = keras.layers.Flatten(name="Flatten")
        self.dense = keras.layers.Dense(1, name="OutputDense")

    def call(self,
             x,
             training=False,
             mask=None):
        x = tf.nn.leaky_relu(self.conv1(x))
        if training:
            x = tf.nn.dropout(x, 0.3)

        x = tf.nn.relu(self.conv2(x))
        if training:
            x = tf.nn.dropout(x, 0.3)

        x = self.flatten(x)
        return self.dense(x)


# %%

class GeneratorFromGithub(AbstractGenerator):

    def __init__(self,
                 name: str = 'github_generator'):
        super(GeneratorFromGithub, self).__init__(name=name)

        self.dense = keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,), name="Dense")
        self.bn0 = keras.layers.BatchNormalization(name="BatchNorm0")

        self.reshape_lyr = keras.layers.Reshape((7, 7, 256), name="Reshape1")

        self.conv2dt1 = keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same",
                                                     use_bias=False, name="Con2DTranspose1")
        self.bn1 = keras.layers.BatchNormalization(name="BatchNorm1")

        self.conv2dt2 = keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same",
                                                     use_bias=False, name="Con2DTranspose2")
        self.bn2 = keras.layers.BatchNormalization(name="BatchNorm2")

        self.conv2dt3 = keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same",
                                                     use_bias=False, activation="tanh", name="Con2DTranspose3")

    def call(self,
             x,
             training=False,
             mask=None):
        x = self.dense(x)
        x = tf.nn.leaky_relu(self.bn0(x))

        x = self.reshape_lyr(x)

        x = self.conv2dt1(x)
        x = tf.nn.leaky_relu(self.bn1(x))

        x = self.conv2dt2(x)
        x = tf.nn.leaky_relu(self.bn2(x))

        x = self.conv2dt3(x)

        return x


# %%


class OneDimGenerator(AbstractGenerator):

    def __init__(self,
                 name: str = 'one_dim_generator',
                 layer_size_list: tuple = (8, 8, 8),
                 output_dim: int = 1,
                 last_layer_activation: str = 'linear'  # softmax?
                 ):
        super(OneDimGenerator, self).__init__(name=name)
        self.layer_size_list = layer_size_list
        self.output_dim = output_dim

        self.layer_list = []
        for layer_size in layer_size_list:
            self.layer_list.append(Dense(layer_size, activation='relu'))  # LeakyReLU?

        self.the_last_layer = Dense(self.output_dim, activation=last_layer_activation)

    def call(self,
             inputs,
             training=True,
             mask=None):

        x = inputs
        for layer_ in self.layer_list:
            x = layer_(x)

        x = self.the_last_layer(x)

        return x


# %%

class OneDimDiscriminator(AbstractDiscriminator):

    def __init__(self,
                 name: str = 'one_dim_discriminator',
                 layer_size_list: tuple = (8, 8, 8),
                 output_dim: int = 1,
                 last_layer_activation: str = 'linear'
                 ):
        super(OneDimDiscriminator, self).__init__(name=name)

        self.layer_size_list = layer_size_list
        self.output_dim = output_dim

        self.layer_list = []
        for layer_size in layer_size_list:
            self.layer_list.append(Dense(layer_size, activation='relu'))  # LeakyReLU?

        self.the_last_layer = Dense(self.output_dim, activation=last_layer_activation)

    def call(self,
             inputs,
             training=True,
             mask=None):

        x = inputs
        for layer_ in self.layer_list:
            x = layer_(x)

        x = self.the_last_layer(x)

        return x


# %%

class AdversarialNetwork(Model):

    def __init__(self,
                 generator: AbstractGenerator,
                 discriminator: AbstractDiscriminator,
                 latent_dim: int,
                 model_name: str = 'adversarial_network',
                 batch_size: int = 100,  # 32
                 ):
        super(AdversarialNetwork, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.model_name = model_name
        self.batch_size = batch_size

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

    # @tf.function( input_signature=[tf.TensorSpec(shape=(self.batch_size, sdfsdfsdf), dtype=tf.float32)])
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

def named_logs(model, logs):
    result = {}
    for log in zip(model.metrics_names, logs):
        result[log[0]] = log[1]
    return result


def loss_fn(labels, output):
    return keras.losses.BinaryCrossentropy(from_logits=True)(labels, output)


# %%

class PrepareTrainData:

    def __init__(self,
                 dataset_name: str = 'mnist',  # one_dim
                 path: Path = Path(__file__).parent.absolute(),
                 buffer_size: int = 60000,  # 10000
                 batch_size: int = 32,
                 ):

        self.dataset_path = os.path.join(path, 'data', dataset_name)
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        if dataset_name == 'mnist':
            self.x_train = self._prepare_mnist()
        elif dataset_name == 'one_dim':
            self.x_train = self._prepare_distribution()

    def _prepare_mnist(self):

        (train_images, _,), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5
        x_train = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self.buffer_size).batch(self.batch_size)

        tf.data.experimental.save(x_train, self.dataset_path)

        return x_train

    def _prepare_distribution(self):

        init_sampling = np.random.normal(0, 1, size=self.buffer_size).astype('float32')
        x_train = init_sampling[(init_sampling < -3) |
                                ((-2 < init_sampling) & (init_sampling < -1)) |
                                ((0 < init_sampling) & (init_sampling < 1)) |
                                ((2 < init_sampling) & (init_sampling < 3))]
        x_train = np.random.choice(x_train, self.buffer_size)[:, np.newaxis]
        x_train = tf.data.Dataset.from_tensor_slices(x_train).batch(self.batch_size)
        tf.data.experimental.save(x_train, self.dataset_path)

        return x_train


#%%

class SaveResultsCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 saving_period: int = 5,
                 saving_path: Path = Path(__file__).parent.absolute()
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


# %%

def testing_gan(x_train,
                generator: AbstractGenerator,
                discriminator: AbstractDiscriminator,
                gan_home_path: Path = Path(__file__).parent.absolute(),
                gen_optimizer: tf.keras.optimizers = None,
                dis_optimizer: tf.keras.optimizers = None,
                latent_dim: int = 100,
                batch_size: int = 32,
                epochs: int = 2,
                experiment: str = '',
                callbacks: tf.keras.callbacks = None
                ):

    directory = os.path.join(gan_home_path, experiment)
    folder_adv_log = os.path.join(directory, 'logs/adv_callback')
    folder_dis_log = os.path.join(directory, 'logs/dis_callback')
    folder_img = os.path.join(directory, 'img/')
    os.makedirs(folder_adv_log, exist_ok=True)
    os.makedirs(folder_dis_log, exist_ok=True)
    os.makedirs(folder_img, exist_ok=True)

    gan = AdversarialNetwork(generator, discriminator, latent_dim, batch_size=batch_size)
    gan.compile(gen_optimizer, dis_optimizer, loss_fn)

    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    tensorboard_callback = TensorBoard(log_dir=os.path.join(gan_home_path, "logs", current_time))

    callbacks.append(tensorboard_callback)

    _history = gan.fit(x_train, epochs=epochs, callbacks=callbacks)


# %%

if __name__ == '__main__':
    # prepare optimizers for 'github'
    github_gen_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.05)
    github_dis_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.05)
    # prepare optimizers for 'book'
    book_gen_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.05)
    lr_ratio = 0.5
    decay_ratio = 0.5
    book_dis_opt = RMSprop(learning_rate=(2e-4 * lr_ratio), decay=(6e-8 * decay_ratio))
    # prepare optimizers for 'one-dim'
    one_dim_gen_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.05)
    one_dim_dis_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.05)
    # prepare train images
    img_data = PrepareTrainData(dataset_name='mnist', batch_size=32).x_train
    # prepare train distributions
    one_dim_data = PrepareTrainData(dataset_name='one_dim', batch_size=100).x_train
    # prepare root directory
    dir = Path(__file__).parent.absolute()
    # prepare callbacks
    callbacks = []

    EXPERIMENT = 'one_dim'

    if EXPERIMENT == 'github':
        gen = GeneratorFromGithub()
        dis = DiscriminatorFromGithub()
        experiment_dir = Path.joinpath(dir, EXPERIMENT)
        testing_gan(x_train=img_data, generator=gen, discriminator=dis, gan_home_path=experiment_dir,
                    gen_optimizer=github_gen_opt, dis_optimizer=github_dis_opt,
                    latent_dim=100, batch_size=32, epochs=2, callbacks=callbacks)

    if EXPERIMENT == 'book':
        gen = ImageGeneratorFromBook()
        dis = ImageDiscriminatorFromBook()
        experiment_dir = Path.joinpath(dir, EXPERIMENT)
        saving_callback = SaveResultsCallback(saving_path=experiment_dir)
        callbacks.append(saving_callback)
        testing_gan(x_train=img_data, generator=gen, discriminator=dis, gan_home_path=experiment_dir,
                    gen_optimizer=book_gen_opt, dis_optimizer=book_dis_opt,
                    latent_dim=100, batch_size=32, epochs=2, callbacks=callbacks)

    if EXPERIMENT == 'one_dim':
        gen = OneDimGenerator()
        dis = OneDimDiscriminator()
        experiment_dir = Path.joinpath(dir, EXPERIMENT)
        saving_callback = SaveResultsCallback(saving_path=experiment_dir)
        callbacks.append(saving_callback)
        testing_gan(x_train=one_dim_data, generator=gen, discriminator=dis, gan_home_path=experiment_dir,
                    gen_optimizer=one_dim_gen_opt, dis_optimizer=one_dim_dis_opt,
                    latent_dim=1, batch_size=100, epochs=2000, callbacks=callbacks)
