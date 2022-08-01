from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, \
                                    Conv2DTranspose, Reshape, Activation

from .abstract_models import AbstractGenerator


class Generator(AbstractGenerator):
    def __init__(self,
                 model_name: str = 'default_generator',
                 gan_type: str = 'dc_gan',
                 image_size: int = 28,
                 kernel_size: int = 5,
                 layer_filters: tuple = (128, 64, 32, 1),
                 layer_strides: tuple = (2, 2, 1, 1),
                 leaky_relu_alpha: float = 0.2,
                 output_activation: str = 'tanh',
                 one_hot_labels: list = None,
                 codes: int = None
                 ):
        super(Generator, self).__init__(name=model_name)
        self.gan_type = gan_type
        self.image_size = image_size
        self.image_resize = image_size // 4
        self.kernel_size = kernel_size
        self.layer_filters = layer_filters
        self.layer_strides = layer_strides
        self.leaky_relu_alpha = leaky_relu_alpha
        self.one_hot_labels = one_hot_labels
        if one_hot_labels:
            self.one_hot_dim = len(one_hot_labels)
        self.codes = codes

        self.generator_blocks = None

        self.output_dense = None
        self.output_activation = Activation(output_activation)
        self.input_dense = None
        self.input_reshape = None

        self.generator_construct = {
            'dc_gan': self._build_default_structure,
            'cond_gan': self._build_default_structure,
            'ls_gan': self._build_default_structure,
            'one_dim': self._build_one_dim_structure
        }

        self.generator_calls = {
            'dc_gan': self._call_default_generator,
            'cond_gan': self._call_default_generator,
            'ls_gan': self._call_default_generator,
            'one_dim': self._call_one_dim_generator
        }

        self.generator_construct[self.gan_type]()

    def call(self, inputs, training=None, mask=None):
        output_ = self.generator_calls[self.gan_type](inputs, training=None, mask=None)
        return output_

    def _build_default_structure(self):
        self.generator_blocks = {}

        for index, filters in enumerate(self.layer_filters):
            strides = self.layer_strides[index]
            self.generator_blocks[f'block_{index}'] = [
                BatchNormalization(name=f'batch_norm_{index}'),
                LeakyReLU(alpha=self.leaky_relu_alpha),
                Conv2DTranspose(filters=filters,
                                kernel_size=self.kernel_size,
                                strides=strides,
                                padding='same',
                                name=f'conv_{index}')
            ]

        self.input_dense = Dense(self.image_resize * self.image_resize * self.layer_filters[0])
        self.input_reshape = Reshape((self.image_resize, self.image_resize, self.layer_filters[0]))

    def _build_one_dim_structure(self):
        self.generator_blocks = []

        for filters in self.layer_filters:
            self.generator_blocks.append(Dense(filters, activation=LeakyReLU(alpha=self.leaky_relu_alpha)))

        self.output_dense = Dense(1)

    def _call_default_generator(self, x, training=None, mask=None):
        x = self.input_dense(x)
        x = self.input_reshape(x)

        for index, filters in enumerate(self.layer_filters):
            for block_ in self.generator_blocks[f"block_{index}"]:
                x = block_(x)

        x = self.output_activation(x)

        return x

    def _call_one_dim_generator(self, x, training=None, mask=None):
        for block_ in self.generator_blocks:
            x = block_(x)

        x = self.output_dense(x)
        x = self.output_activation(x)

        return x
