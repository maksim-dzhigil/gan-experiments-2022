from tensorflow.keras.layers import Dense, Conv2D, Flatten, \
                                    Activation, LeakyReLU

from .abstract_models import AbstractDiscriminator


class Discriminator(AbstractDiscriminator):
    def __init__(self,
                 model_name: str = 'default_discriminator',
                 gan_type: str = 'dc_gan',
                 kernel_size: int = 5,
                 layer_filters: tuple = (32, 64, 128, 256),
                 layer_strides: tuple = (2, 2, 2, 1),
                 leaky_relu_alpha: float = 0.2,
                 output_activation: str = 'sigmoid',
                 one_hot_labels: list = None,
                 num_codes: int = None
                 ):
        super(Discriminator, self).__init__(name=model_name)
        self.gan_type = gan_type
        self.kernel_size = kernel_size
        self.layer_filters = layer_filters
        self.layer_strides = layer_strides
        self.leaky_relu_alpha = leaky_relu_alpha
        self.output_activation = output_activation
        self.one_hot_labels = one_hot_labels
        if one_hot_labels:
            self.one_hot_labels_dim = len(one_hot_labels)
        self.num_codes = num_codes

        self.discriminator_blocks = None

        self.flatten_ = None
        self.output_dense = None
        self.output_activation = Activation(output_activation)

        self.discriminator_construct = {
            'dc_gan': self._build_default_structure,
            'cond_gan': self._build_default_structure,
            'ls_gan': self._build_default_structure,
            'one_dim': self._build_one_dim_structure
        }

        self.discriminator_calls = {
            'dc_gan': self._call_default_discriminator,
            'cond_gan': self._call_default_discriminator,
            'ls_gan': self._call_default_discriminator,
            'one_dim': self._call_one_dim_discriminator
        }

        self.discriminator_construct[self.gan_type]()

    def call(self, inputs, training=None, mask=None):
        output_ = self.discriminator_calls[self.gan_type](inputs, training=None, mask=None)
        return output_

    def _build_default_structure(self):
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
        self.output_dense = Dense(1)

    def _build_one_dim_structure(self):
        self.discriminator_blocks = []

        for filters in self.layer_filters:
            self.discriminator_blocks.append(Dense(filters, activation=LeakyReLU(alpha=self.leaky_relu_alpha)))

        self.output_dense = Dense(1)

    def _call_default_discriminator(self, x, training=None, mask=None):
        for index, filters in enumerate(self.layer_filters):
            for block_ in self.discriminator_blocks[f'block_{index}']:
                x = block_(x)
        x = self.flatten_(x)
        x = self.output_dense(x)
        x = self.output_activation(x)

        return x

    def _call_one_dim_discriminator(self, x, training=None, mask=None):
        for block_ in self.discriminator_blocks:
            x = block_(x)

        x = self.output_dense(x)
        x = self.output_activation(x)

        return x
