from tensorflow.keras.layers import Dense, Conv2D, Flatten
import tensorflow as tf

from abstract_models import AbstractDiscriminator


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
