from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, \
                                    Conv2DTranspose, Reshape, Activation

from .abstract_models import AbstractGenerator


class GanGenerator(AbstractGenerator):
    def __init__(self,
                 model_name: str = 'generator_dcgan',
                 image_size: int = 28,
                 kernel_size: int = 5,
                 layer_filters: tuple = (128, 64, 32, 1),
                 layer_strides: tuple = (2, 2, 1, 1),
                 leaky_relu_alpha: float = 0.2,
                 output_activation: str = 'tanh',):
        super(GanGenerator, self).__init__(name=model_name)
        self.image_size = image_size
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

    # @tf.autograph.experimental.do_not_convert
    def call(self,
             x,
             training=None,
             mask=None):
        x = self.input_dense(x)
        x = self.input_reshape(x)

        for index, filters in enumerate(self.layer_filters):
            for block_ in self.generator_blocks[f"block_{index}"]:
                x = block_(x)

        x = self.output_activation(x)

        return x
