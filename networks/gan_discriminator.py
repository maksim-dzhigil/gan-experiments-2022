from tensorflow.keras.layers import Dense, Conv2D, Flatten, \
                                    Activation, LeakyReLU

from abstract_models import AbstractDiscriminator


class GanDiscriminator(AbstractDiscriminator):
    def __init__(self,
                 model_name: str = 'discriminator_for_dcgan',
                 kernel_size: int = 5,
                 layer_filters: tuple = (32, 64, 128, 256),
                 layer_strides: tuple = (2, 2, 2, 1),
                 leaky_relu_alpha: float = 0.2,
                 output_activation: str = 'sigmoid'):
        super(GanDiscriminator, self).__init__(name=model_name)
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
                       name=f'conv_{index}')
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
        x = self.output_activation(x)

        return x
