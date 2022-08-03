import unittest

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam, RMSprop
from datetime import datetime
from networks.discriminator import Discriminator
from networks.generator import Generator
from networks.github_generator import GithubGenerator
from networks.github_discriminator import GithubDiscriminator
from networks.adversarial_models import AdversarialNetwork, ConditionalAdversarialNetwork
from tools.save_results_callback import SaveResultsCallback
from tools.utility_functions import binary_cross_entropy
from train_data.prepare_train_data import PrepareTrainData
from os import path

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_buffer_size(self):
        num_of_elements = 10
        batch_size = 1
        ptd = PrepareTrainData(number_of_elements=num_of_elements, batch_size=batch_size)
        self.assertEqual(num_of_elements, len(list(ptd.dataset)))

    def test_integration_one_dim_gan(self):
        gen_optimizer = 'adam'
        dis_optimizer = 'rmsprop'
        batch_size = 100
        buffer_size = 10000
        latent_dim = 100
        epochs = 1
        dis_layer_filters: tuple = (8, 8, 8)
        gen_layer_filters: tuple = (8, 8, 8)

        generator = Generator(gan_type='one_dim',
                              layer_filters=gen_layer_filters)

        discriminator = Discriminator(gan_type='one_dim',
                                      layer_filters=dis_layer_filters)

        optimizers = {
            'adam': Adam(),
            'rmsprop': RMSprop()
        }

        ptd = PrepareTrainData(dataset_name='one_dim',
                               buffer_size=buffer_size,
                               batch_size=batch_size)
        train_data = ptd.dataset.take(1)

        generator_optimizer = optimizers[gen_optimizer]
        discriminator_optimizer = optimizers[dis_optimizer]

        gan = AdversarialNetwork(generator, discriminator,
                                 latent_dim=latent_dim,
                                 batch_size=batch_size,
                                 data_nature='one_dim')
        gan.compile(generator_optimizer, discriminator_optimizer, binary_cross_entropy)  # add loss_fn settings?

        _history = gan.fit(train_data, epochs=epochs)

    def test_integration_ls_gan(self):

        gen_optimizer = 'adam'
        dis_optimizer = 'rmsprop'
        epochs = 1
        buffer_size = 1
        batch_size = 1

        generator = Generator()
        discriminator = Discriminator()

        optimizers = {
            'adam': Adam(),
            'rmsprop': RMSprop()
        }

        ptd = PrepareTrainData(buffer_size=buffer_size,
                               batch_size=batch_size)
        train_data = ptd.dataset.take(1)

        generator_optimizer = optimizers[gen_optimizer]
        discriminator_optimizer = optimizers[dis_optimizer]

        gan = AdversarialNetwork(generator, discriminator, batch_size=batch_size)
        gan.compile(generator_optimizer, discriminator_optimizer, binary_cross_entropy)

        _history = gan.fit(train_data, epochs=epochs)

    def test_integration_github_gan(self):

        epochs = 1
        buffer_size = 1
        batch_size = 1

        generator = GithubGenerator()
        discriminator = GithubDiscriminator()

        optimizers = {'adam': Adam(), 'rmsprop': RMSprop()}

        ptd = PrepareTrainData(buffer_size=buffer_size,
                               batch_size=batch_size)

        train_data = ptd.dataset.take(1)

        generator_optimizer = optimizers['adam']
        discriminator_optimizer = optimizers['adam']

        gan = AdversarialNetwork(generator, discriminator,
                                 batch_size=batch_size, data_nature='mnist')
        gan.compile(generator_optimizer, discriminator_optimizer, binary_cross_entropy)

        _history = gan.fit(train_data, epochs=epochs)

    def test_integration_dc_gan(self):
        generator = Generator()
        discriminator = Discriminator()
        buffer_size = 1
        batch_size = 1
        epochs = 1

        optimizers = {'adam': Adam(), 'rmsprop': RMSprop()}

        ptd = PrepareTrainData(buffer_size=buffer_size,
                               batch_size=batch_size)

        train_data = ptd.dataset.take(1)

        generator_optimizer = optimizers['adam']
        discriminator_optimizer = optimizers['adam']

        gan = AdversarialNetwork(generator, discriminator,
                                 batch_size=batch_size, data_nature='mnist')
        gan.compile(generator_optimizer, discriminator_optimizer, binary_cross_entropy)
        _history = gan.fit(train_data, epochs=epochs)

    def test_integration_cond_gan(self):
        generator = Generator(gan_type='cond_gan')
        discriminator = Discriminator(gan_type='cond_gan')
        buffer_size = 10000
        batch_size = 1
        epochs = 1

        ptd = PrepareTrainData(buffer_size=buffer_size,
                               batch_size=batch_size)

        train_data = ptd.dataset.take(1)

        optimizers = {'adam': Adam(), 'rmsprop': RMSprop()}

        generator_optimizer = optimizers['adam']
        discriminator_optimizer = optimizers['adam']

        gan = ConditionalAdversarialNetwork(generator, discriminator,
                                            batch_size=batch_size)
        gan.compile(generator_optimizer, discriminator_optimizer, binary_cross_entropy)
        _history = gan.fit(train_data, epochs=epochs)


if __name__ == '__main__':
    unittest.main()
