from abstract_models import AbstractDiscriminator
from tensorflow.keras.layers import Dense


class OneDimDiscriminator(AbstractDiscriminator):
    def __init__(self,
                 model_name: str = 'one_dim_discriminator',
                 layer_filters: tuple = (8, 8, 8),
                 output_activation: str = 'linear'):
        super(OneDimDiscriminator, self).__init__(name=model_name)
        self.layer_filters = layer_filters

        self.discriminator_blocks = []

        for filters in self.layer_filters:
            self.discriminator_blocks.append(Dense(filters, activation='relu'))  # LeakyReLU?

        self.output_ = Dense(1, activation=output_activation)

    def call(self,
             x,
             training=False,
             mask=None):
        for block_ in self.discriminator_blocks:
            x = block_(x)

        x = self.output_(x)

        return x

