from abstract_models import AbstractGenerator
from tensorflow.keras.layers import Dense


class OneDimGenerator(AbstractGenerator):
    def __init__(self,
                 model_name: str = 'one_dim_generator',
                 layer_filters: tuple = (8, 8, 8),
                 output_activation: str = 'linear'  # softmax?
                 ):
        super(OneDimGenerator, self).__init__(name=model_name)
        self.layer_filters = layer_filters

        self.generator_blocks = []

        for filters in self.layer_filters:
            self.generator_blocks.append(Dense(filters, activation='relu'))  # LeakyReLU?

        self.output_ = Dense(1, activation=output_activation)

    def call(self,
             x,
             training=None,
             mask=None):
        for block_ in self.generator_blocks:
            x = block_(x)

        x = self.output_(x)

        return x
