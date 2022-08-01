from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam, RMSprop
from datetime import datetime
from networks.gan_discriminator import GanDiscriminator
from networks.generator import Generator
from networks.adversarial_models import ConditionalAdversarialNetwork
from tools.save_results_callback import SaveResultsCallback
from tools.utility_functions import binary_cross_entropy
from train_data.prepare_train_data import PrepareTrainData
from os import path


def conduct_experiment_cond_gan(gen_optimizer: str = 'adam',
                                dis_optimizer: str = 'rmsprop',
                                adam_lr: float = 2e-4,
                                adam_beta_1: float = 0.05,
                                adam_beta_2: float = 0.999,
                                rms_lr: float = 2e-4,
                                rms_decay: float = 6e-8,
                                rms_lr_ratio: float = 0.5,
                                rms_decay_ratio: float = 0.5,
                                batch_size: int = 32,
                                buffer_size: int = 60000,
                                latent_dim: int = 100,
                                epochs: int = 40000,
                                path_to_root: path = path.dirname(__file__),
                                saving_period: int = 500,
                                dis_kernel_size: int = 5,
                                dis_layer_filters: tuple = (32, 64, 128, 256),
                                dis_layer_strides: tuple = (2, 2, 2, 1),
                                dis_leaky_relu_alpha: float = 0.2,
                                dis_output_activation: str = 'sigmoid',
                                gen_image_size: int = 28,
                                gen_kernel_size: int = 5,
                                gen_layer_filters: tuple = (128, 64, 32, 1),
                                gen_layer_strides: tuple = (2, 2, 1, 1),
                                gen_leaky_relu_alpha: float = 0.2,
                                gen_output_activation: str = 'tanh'):

    directory = path.join(path_to_root, 'cond_gan')

    generator = Generator(model_name='conditional_gan',
                          gan_type='cond_gan',
                          image_size=gen_image_size,
                          kernel_size=gen_kernel_size,
                          layer_filters=gen_layer_filters,
                          layer_strides=gen_layer_strides,
                          leaky_relu_alpha=gen_leaky_relu_alpha,
                          output_activation=gen_output_activation,
                          )

    discriminator = GanDiscriminator(kernel_size=dis_kernel_size,
                                     layer_filters=dis_layer_filters,
                                     layer_strides=dis_layer_strides,
                                     leaky_relu_alpha=dis_leaky_relu_alpha,
                                     output_activation=dis_output_activation)

    callbacks = []
    saving_callback = SaveResultsCallback(saving_period=saving_period,
                                          saving_path=directory)

    optimizers = {
        'adam': Adam(learning_rate=adam_lr,
                     beta_1=adam_beta_1,
                     beta_2=adam_beta_2),
        'rmsprop': RMSprop(learning_rate=rms_lr * rms_lr_ratio,
                           decay=rms_decay * rms_decay_ratio,
                           )
    }

    ptd = PrepareTrainData(dataset_name='mnist',
                           path_to_root=directory,
                           buffer_size=buffer_size,
                           batch_size=batch_size)
    dataset = ptd.dataset

    generator_optimizer = optimizers[gen_optimizer]
    discriminator_optimizer = optimizers[dis_optimizer]

    gan = ConditionalAdversarialNetwork(generator, discriminator, latent_dim=latent_dim, batch_size=batch_size)
    gan.compile(generator_optimizer, discriminator_optimizer, binary_cross_entropy)  # add loss_fn settings?

    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    tensorboard_callback = TensorBoard(log_dir=path.join(directory, "logs", current_time))

    callbacks.append(saving_callback)
    callbacks.append(tensorboard_callback)

    _history = gan.fit(dataset, epochs=epochs, callbacks=callbacks)


if __name__ == '__main__':
    conduct_experiment_cond_gan()
