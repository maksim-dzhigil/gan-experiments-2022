import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization

from .abstract_models import AbstractGenerator


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
