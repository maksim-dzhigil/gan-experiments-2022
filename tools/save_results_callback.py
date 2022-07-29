from tensorflow.keras.callbacks import Callback
from os import path, makedirs
from math import sqrt
import matplotlib.pyplot as plt
from numpy import zeros

import tensorflow as tf  # only for tf.random.normal (29)


class SaveResultsCallback(Callback):
    def __init__(self,
                 saving_period: int = 500,
                 saving_path: path = path.dirname(__file__)):
        super(SaveResultsCallback, self).__init__()
        self.saving_period = saving_period
        self.images_saving_path = path.join(saving_path, 'img')
        self.models_saving_path = path.join(saving_path, 'models')

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.saving_period == 0:
            self._save_images(epoch)
            self._save_models(epoch)

    def _save_images(self, epoch: int):
        if not path.exists(self.images_saving_path):
            makedirs(self.images_saving_path)
        image_name = path.join(self.images_saving_path, '%05d' % (epoch + 1))
        random_latent_vectors = tf.random.normal(shape=(self.model.batch_size, self.model.latent_dim))
        generated_images = self.model.generator.predict(random_latent_vectors)

        if generated_images.ndim == 2:
            plt.scatter(generated_images[:, 0], zeros(generated_images.shape[0]))
            plt.xlim([-4, 4])
            plt.savefig(image_name)
            plt.close('all')

        elif generated_images.ndim == 4:
            plt.figure(figsize=(2.2, 2.2))
            rows = int(sqrt(random_latent_vectors.shape[0]))
            for i in range(generated_images.shape[0]):
                plt.subplot(rows, rows, i+1)
                plt.imshow(generated_images[i, :, :, 0], cmap='gray')
                plt.axis('off')
            plt.savefig(image_name)
            plt.close('all')

    def _save_models(self, epoch: int):
        if not path.exists(self.models_saving_path):
            makedirs(self.models_saving_path)
        discriminator_path = path.join(self.models_saving_path, '%05d_discriminator' % (epoch + 1))
        generator_path = path.join(self.models_saving_path, '%05d_generator' % (epoch + 1))

        self.model.discriminator.save(discriminator_path, save_format='tf')
        self.model.generator.save(generator_path, save_format='tf')
