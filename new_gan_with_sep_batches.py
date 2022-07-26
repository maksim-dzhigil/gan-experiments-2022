import sys

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

class AbstractDescriminator(Model):

    def __init__(self, *kargs, **kvargs):
        super(AbstractDescriminator, self).__init__(*kargs, **kvargs)

class AbstractGenerator(Model):

    def __init__(self, *kargs, **kvargs):
        super(AbstractGenerator, self).__init__(*kargs, **kvargs)




class PrepareTrainData:

    def __init__(self,
                 dataset_name: str = 'mnist',
                 path: Path = Path(__file__).parent.absolute(),
                 buffer_size: int = 60000,
                 batch_size: int = 100):  # batch_size: int = 32):
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
        batch_size: int = 100 #batch_size: int = 32
):
    train_dataset = DATASET_FUNCTION[dataset](
        data_path = dataset_save_path / "data" / f"{dataset}.npz",
        buffer_size = buffer_size,
        batch_size = batch_size
    )

    return train_dataset

# %%


class ImageDiscriminatorFromBook(AbstractDescriminator):

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

    def call(self,
             x,
             training=False,
             mask=None):

        for index, filters in enumerate(self.layer_filters):
            for layer_ in self.blocks[f"block_{index + 1}"]:
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


class ImageGeneratorFromBook(AbstractGenerator):

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
            strides = self.layer_strides[index]
            self.blocks[f'block_{index + 1}'] = [
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
            for layer_ in self.blocks[f"block_{index + 1}"]:
                x = layer_(x)

        x = self.output_activation(x)

        return x


# %%


class DiscriminatorFromGithub(AbstractDescriminator):
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


#%%


class GeneratorFromGithub(AbstractGenerator):

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


    def call(self,
             x,
             training=False,
             mask=None):
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
                 model_name: str = 'adversarial_network',
                 batch_size: int = 100 #32
                 ):
        super(AdversarialNetwork, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.model_name = model_name
        self.batch_size = batch_size
        print("BS:", self.batch_size)

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


    #@tf.function( input_signature=[tf.TensorSpec(shape=(self.batch_size, sdfsdfsdf), dtype=tf.float32)])
    @tf.function
    def train_step(self, data):
        print('*****start*****')
        real_images = data
        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))

        generated_images = self.generator(random_latent_vectors)
        combined_images = tf.concat([generated_images, real_images], axis=0)
        #tf.print(combined_images.shape, output_stream=sys.stderr)


        print("fsdfdsfsd ", generated_images.shape, real_images.shape)
        print("real ", real_images.shape)
        print("gener ", generated_images.shape)
        labels = tf.concat([tf.ones((self.batch_size, 1)),
                            tf.zeros((self.batch_size, 1))], axis=0)

        print("labels ", labels.shape)
        print("combined ", combined_images.shape)

        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        misleading_labels = tf.zeros((self.batch_size, 1))

        print('***still_fine****')

        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        print('*****end*****')
        return {"d_loss": d_loss, "g_loss": g_loss}


#%%


def loss_fn(labels, output):
    return keras.losses.BinaryCrossentropy(from_logits=True)(labels, output)


#%%


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

        self._create_callback()


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


#%%


def testing_github_gan(x_train,
                       generator : AbstractGenerator,
                       discriminator: AbstractDescriminator,
                       path: Path = Path('./GAN_with_sep_batches'),
                       latent_dim=100,
                       batch_size=100): # batch_size=32):

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.05)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.05)

    save_models = True
    epochs = 2

    gan = AdversarialNetwork(generator, discriminator, latent_dim, batch_size=batch_size)
    gan.compile(discriminator_optimizer, generator_optimizer, loss_fn)

    gan_home_path = path
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    callbacks = []
    tensorboard_callback = TensorBoard(log_dir=os.path.join(gan_home_path, "logs", current_time))

    callbacks.append(tensorboard_callback)

    if save_models:
        checkpoint_filepath = Path.joinpath(gan_home_path, "checkpoints", f"mnist_{epochs}_{current_time}", "checkpoint")
        model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath)
        callbacks.append(model_checkpoint_callback)

    #print("shape", next(iter(dgf))[0].shape)
    _history = gan.fit(x_train, epochs=epochs, callbacks=callbacks)

    generator.save(os.path.join(gan_home_path, "saved_models", current_time, f"generator_mnist_{epochs}"))
    discriminator.save(os.path.join(gan_home_path, "saved_models", current_time, f"discriminator_mnist_{epochs}"))


#%%


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



#%%

if __name__ == '__main__':
    ptd = PrepareTrainData()

    img_train = ptd.x_train
    dgf = DataGeneratorForDenseGAN()
    _, one_dim_train = next(iter(dgf))
    experiment = 'github'

    if experiment == 'github':
        discriminator = DiscriminatorFromGithub()
        generator = GeneratorFromGithub()
        testing_github_gan(img_train, generator, discriminator)
    elif experiment == 'book':
        discriminator = ImageDiscriminatorFromBook()
        generator = ImageGeneratorFromBook()
        testing_github_gan(img_train, generator, discriminator)
    elif experiment == 'one_dim':
        discriminator = OneDimDiscriminatorDense()
        generator = OneDimGeneratorDense()
        testing_github_gan(one_dim_train, generator, discriminator, latent_dim=1, batch_size=100)
