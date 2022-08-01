import tensorflow as tf

from tensorflow.keras.models import Model

from .abstract_models import AbstractGenerator, AbstractDiscriminator


class Adversarial(Model):
    def __init__(self,
                 generator: AbstractGenerator,
                 discriminator: AbstractDiscriminator,
                 latent_dim: int = 100,
                 batch_size: int = 100,
                 model_name: str = 'adversarial',
                 data_nature: str = 'mnist',
                 num_classes: int = 10
                 ):
        super(Adversarial, self).__init__(name=model_name)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.data_nature = data_nature
        self.num_classes = num_classes

        self.d_optimizer = None
        self.g_optimizer = None
        self.loss_function = None

    def compile(self,
                g_optimizer: tf.keras.optimizers = None,
                d_optimizer: tf.keras.optimizers = None,
                loss_function: tf.keras.losses = None
                ):
        super(Adversarial, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_function = loss_function

    @tf.function
    def train_step(self, data):
        if self.data_nature == 'mnist':
            pass
        elif self.data_nature == 'cond_mnist':
            pass
        elif self.data_nature == 'one_dim':
            pass
