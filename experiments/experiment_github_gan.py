from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam, RMSprop
from datetime import datetime
from networks.github_discriminator import GithubDiscriminator
from networks.github_generator import GithubGenerator
from networks.adversarial_models import AdversarialNetwork
from tools.save_results_callback import SaveResultsCallback
from tools.utility_functions import loss_fn
from train_data.prepare_train_data import PrepareTrainData
from os import path


def conduct_experiment_github_gan(gen_optimizer: str = 'adam',
                                  dis_optimizer: str = 'adam',
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
                                  saving_period: int = 500
                                  ):

    directory = path.join(path_to_root, 'github_gan')

    generator = GithubGenerator()
    discriminator = GithubDiscriminator()

    callbacks = []
    saving_callback = SaveResultsCallback(saving_period=saving_period,
                                          saving_path=directory)

    optimizers = {
        'adam': Adam(learning_rate=adam_lr,
                     beta_1=adam_beta_1,
                     beta_2=adam_beta_2),
        'rmsprop': RMSprop(learning_rate=rms_lr * rms_lr_ratio,
                           decay=rms_decay * rms_decay_ratio,)
    }

    ptd = PrepareTrainData(dataset_name='mnist',
                           path_to_root=directory,
                           buffer_size=buffer_size,
                           batch_size=batch_size)
    dataset = ptd.dataset

    generator_optimizer = optimizers[gen_optimizer]
    discriminator_optimizer = optimizers[dis_optimizer]

    gan = AdversarialNetwork(generator, discriminator, latent_dim=latent_dim, batch_size=batch_size)
    gan.compile(generator_optimizer, discriminator_optimizer, loss_fn)

    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    tensorboard_callback = TensorBoard(log_dir=path.join(directory, "logs", current_time))

    callbacks.append(saving_callback)
    callbacks.append(tensorboard_callback)

    _history = gan.fit(dataset, epochs=epochs, callbacks=callbacks)


if __name__ == "__main__":
    conduct_experiment_github_gan()
