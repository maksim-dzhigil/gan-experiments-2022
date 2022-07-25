from matplotlib import pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Dense, Reshape, \
    BatchNormalization, Conv2DTranspose, LeakyReLU, Conv2D, Flatten

from datetime import datetime
from json import loads
from pathlib import Path
import os

from tensorflow.keras.datasets import mnist


# %%


class PrepareTrainData:

    def __init__(self,
                 dataset_name: str = 'mnist',
                 path: Path = Path(__file__).parent.absolute(),
                 buffer_size: int = 60000,
                 batch_size: int = 32):
        self.dataset_path = os.path.join(path, dataset_name, '.npz')
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.x_train = self._prepare_mnist()

    def _prepare_mnist(self):
        (train_images, _), (_, _) = keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5
        x_train = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self.buffer_size).batch(self.batch_size)
        tf.data.experimental.save(x_train, self.dataset_path)
        return x_train


def prepare_mnist(data_path, buffer_size, batch_size):
    (train_images, _), (_, _) = keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5

    return tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)


DATASET_FUNCTION = {
    "mnist": prepare_mnist
}

def load_dataset(
        dataset: str = "mnist",
        dataset_save_path: Path = Path(__file__).parent.absolute(),
        buffer_size: int = 60000,
        batch_size: int = 32
):
    train_dataset = DATASET_FUNCTION[dataset](
        data_path = dataset_save_path / "data" / f"{dataset}.npz",
        buffer_size = buffer_size,
        batch_size = batch_size
    )

    return train_dataset

# %%


class ImageDiscriminatorFromBook(Model):

    def __init__(self,
                 model_name: str = 'discriminator',
                 kernel_size: int = 5,
                 layer_filters: tuple = (32, 64, 128, 256),
                 layer_strides: tuple = (2, 2, 2, 1),
                 leaky_relu_ratio: float = 0.2,
                 output_activation: str = 'softmax',
                 dropout_value: float = None
                 ):
        super(ImageDiscriminatorFromBook, self).__init__()
        self.model_name = model_name
        self.kernel_size = kernel_size
        self.layer_filters = layer_filters
        self.layer_strides = layer_strides
        self.leaky_relu_ratio = leaky_relu_ratio
        self.dropout_value = dropout_value

        self.blocks = {}

        for index, filters in enumerate(self.layer_filters):
            strides = self.strides[index]
            self.bloks[f'block_{index + 1}'] = [
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

    def call(self,
             x,
             training=False,
             mask=None):

        for index, filters in enumerate(self.layer_filters):
            for layer_ in self.block_dict[f"block_{index + 1}"]:
                x = layer_(x)
                if training:
                    x = tf.nn.dropout(x, 0.3)

        x = self.flatten(x)
        x = self.output_(x)
        x = self.output_activation(x)  # don't use?

        return x


# %%
# !!!! в примере с гитхаба активация используется после нормлаизации батча, а у нас наоборот
# + там вообще не используется активация последнего слоя


class ImageGeneratorFromBook(Model):

    def __init__(self,
                 model_name: str = 'generator',
                 image_size: int = 28,
                 kernel_size: int = 5,
                 layer_filters: tuple = (128, 64, 32, 1),
                 layer_strides: tuple = (1, 1, 2, 2),
                 leaky_relu_ratio: float = 0.2,
                 output_activation: str = 'softmax',
                 ):
        super(ImageGeneratorFromBook, self).__init__()

        self.Model_name = model_name
        self.image_resize = image_size // 4
        self.kernel_size = kernel_size
        self.layer_filters = layer_filters
        self.layer_strides = layer_strides
        self.leaky_relu_ratio = leaky_relu_ratio

        self.blocks = {}

        for index, filters in enumerate(self.layer_filters):
            strides = self.strides[index]
            self.bloks[f'block_{index + 1}'] = [
                BatchNormalization(name=f'batch_norm_{index + 1}'),
                LeakyReLU(alpha=self.leaky_relu_ratio),
                Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='same',
                                name=f'conv_{index + 1}')
            ]

        self.input_dense = Dense(self.image_resize * self.image_resize * layer_filters[0])
        self.input_reshape = Reshape((self.image_resize, self.image_resize, layer_filters[0]))
        self.output_activation = Activation(output_activation)

    def call(self,
             x,
             training=False,
             mask=None):

        x = self.input_dense(x)
        x = self.input_reshape(x)

        for index, filters in enumerate(self.layer_filters):
            for layer_ in self.block_dict[f"block_{index + 1}"]:
                x = layer_(x)

        x = self.output_activation(x)

        return x


# %%


class DiscriminatorFromGithub(keras.Model):
    def __init__(self):
        super(DiscriminatorFromGithub, self).__init__()
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


    def call(self, x, training=False):
        x = tf.nn.leaky_relu(self.conv1(x))
        if training:
            x = tf.nn.dropout(x, 0.3)

        x = tf.nn.relu(self.conv2(x))
        if training:
            x = tf.nn.dropout(x, 0.3)

        x = self.flatten(x)
        return self.dense(x)


#%%


class GeneratorFromGithub(keras.Model):

    def __init__(self):
        super(GeneratorFromGithub, self).__init__()

        self.dense = keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,), name="Dense")
        self.bn0 = keras.layers.BatchNormalization(name="BatchNorm0")

        self.rshpe_lyr = keras.layers.Reshape((7, 7, 256), name="Reshape1")

        self.conv2dt1 = keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False, name="Con2DTranspose1")
        self.bn1 = keras.layers.BatchNormalization(name="BatchNorm1")

        self.conv2dt2 = keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False, name="Con2DTranspose2")
        self.bn2 = keras.layers.BatchNormalization(name="BatchNorm2")

        self.conv2dt3 = keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh", name="Con2DTranspose3")


    def call(self, x):
        x = self.dense(x)
        x = tf.nn.leaky_relu(self.bn0(x))

        x = self.rshpe_lyr(x)

        x = self.conv2dt1(x)
        x = tf.nn.leaky_relu(self.bn1(x))

        x = self.conv2dt2(x)
        x = tf.nn.leaky_relu(self.bn2(x))

        x = self.conv2dt3(x)

        return x


#%%


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


#%%


class OneDimGeneratorDense(SimpleDenseModel):

    def __init__(self,
                 name='simple_dense_generator',
                 layer_size_list=[8, 8, 8],
                 output_shape_=1):
        super(OneDimGeneratorDense, self).__init__(name=name,
                                                   layer_size_list=layer_size_list,
                                                   output_shape_=output_shape_,
                                                   last_layer_activation='linear')


#%%


class OneDimDiscriminatorDense(SimpleDenseModel):

    def __init__(self,
                 name='simple_dense_discriminator',
                 layer_size_list=[8, 8, 8]):
        super(OneDimDiscriminatorDense, self).__init__(name=name,
                                                       layer_size_list=layer_size_list,
                                                       output_shape_=1,
                                                       last_layer_activation='sigmoid')


#%%


class AdversarialNetwork(Model):

    def __init__(self,
                 generator: Model,
                 discriminator: Model,
                 latent_dim: int,
                 model_name: str = 'adversarial_network'
                 ):
        super(AdversarialNetwork, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.model_name = model_name

        self.d_optimizer = None
        self.g_optimizer = None
        self.loss = None

    def compile(self,
                d_optimizer: tf.keras.optimizers = None,
                g_optimizer: tf.keras.optimizers = None,
                loss_fn: tf.keras.losses = None
                ):
        super(AdversarialNetwork, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, data):

        real_images = data
        batch_size = 32  # то самое проблемное место
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        generated_images = self.generator(random_latent_vectors)
        combined_images = tf.concat([generated_images, real_images], axis=0)

        labels = tf.concat([tf.ones((batch_size, 1)),
                            tf.zeros((batch_size, 1))], axis=0)

        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}


#%%


def loss_fn(labels, output):
    return keras.losses.BinaryCrossentropy(from_logits=True)(labels, output)


def prepare_data():
    ptd = PrepareTrainData()
    return ptd.x_train


#%%


def testing_github_gan(x_train, path: Path = None):

    if not path:
        train_dataset = load_dataset(dataset='mnist', buffer_size=60000, batch_size=32)
    else:
        train_dataset = load_dataset(dataset='mnist', dataset_save_path=Path(path),
                                     buffer_size=60000, batch_size=32)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.05)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.05)

    latent_dim = 100
    save_models = True
    epochs = 2

    discriminator = DiscriminatorFromGithub()
    generator = GeneratorFromGithub()
    gan = AdversarialNetwork(generator, discriminator, latent_dim)
    gan.compile(discriminator_optimizer, generator_optimizer, loss_fn)

    gan_home_path = Path('./GAN_with_sep_batches')
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    callbacks = []
    tensorboard_callback = TensorBoard(log_dir=os.path.join(gan_home_path, "logs", current_time))

    callbacks.append(tensorboard_callback)

    if save_models:
        checkpoint_filepath = Path.joinpath(gan_home_path, "checkpoints", f"mnist_{epochs}_{current_time}", "checkpoint")
        model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath)
        callbacks.append(model_checkpoint_callback)

    _history = gan.fit(x_train, epochs=epochs, callbacks=callbacks)

    generator.save(os.path.join(gan_home_path, "saved_models", current_time, f"generator_mnist_{epochs}"))
    discriminator.save(os.path.join(gan_home_path, "saved_models", current_time, f"discriminator_mnist_{epochs}"))


#%%


def testing_gan_from_book(x_train):

    name = 'basic'
    dis = ImageDiscriminatorFromBook(name=name + "_dis")
    gen = ImageGeneratorFromBook(name=name + "_gen")



#%%


def testing_one_dim_gan():
    pass


#%%

if __name__ == '__main__':
    ptd = PrepareTrainData()
    x_train = ptd.x_train
    testing_github_gan(x_train)
