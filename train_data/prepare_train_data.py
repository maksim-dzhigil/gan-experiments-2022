import tensorflow as tf
from numpy import random, newaxis
from os import path


class PrepareTrainData:
    def __init__(self,
                 dataset_name: str = 'mnist',
                 path_to_root: path = path.dirname(__file__),
                 number_of_elements: int = 60000,
                 buffer_size: int = 60000,
                 batch_size: int = 32):
        super(PrepareTrainData, self).__init__()
        self.dataset_name = dataset_name
        self.dataset_path = path.join(path_to_root, 'data', dataset_name)
        self.number_of_elements = number_of_elements
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        datasets = {
            'mnist': self._prepare_mnist(),
            'one_dim': self._prepare_distribution()
        }

        self.dataset = datasets[dataset_name]

    def _prepare_mnist(self):
        (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
        x_train = (x_train - 127.5) / 127.5
        # x_train = x_train / 255.0

        y_train = tf.keras.utils.to_categorical(y_train)

        dataset = tf.data.Dataset \
            .from_tensor_slices((x_train, y_train)) \
            .shuffle(self.buffer_size) \
            .padded_batch(self.batch_size)

        # tf.data.experimental.save(dataset, self.dataset_path)

        return dataset

    def _prepare_distribution(self):
        init_sampling = random.normal(0, 1, size=self.number_of_elements).astype('float32')
        x_train = init_sampling[(init_sampling < -3) |
                                ((-2 < init_sampling) & (init_sampling < -1)) |
                                ((0 < init_sampling) & (init_sampling < 1)) |
                                ((2 < init_sampling) & (init_sampling < 3))]
        x_train = random.choice(x_train, self.number_of_elements)[:, newaxis]

        dataset = tf.data.Dataset \
            .from_tensor_slices(x_train) \
            .batch(self.batch_size)

        # tf.data.experimental.save(dataset, self.dataset_path)

        return dataset
