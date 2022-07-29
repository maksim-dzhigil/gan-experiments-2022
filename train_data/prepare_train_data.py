import tensorflow as tf
from numpy import random, newaxis
from os import path


class PrepareTrainData:
    def __init__(self,
                 dataset_name: str = 'mnist',
                 path_to_root: path = path.dirname(__file__),
                 buffer_size: int = 10000,
                 batch_size: int = 100):
        self.dataset_name = dataset_name
        self.dataset_path = path.join(path_to_root, 'data', dataset_name)
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        datasets = {
            'mnist': self._prepare_mnist(),
            'one_dim': self._prepare_distribution()
        }

        self.dataset = datasets[dataset_name]

    def _prepare_mnist(self):
        # if os.path.exists(self.dataset_path):
        # The function is deprecated. Use tf.data.Dataset.load() instead
        # return tf.data.experimental.load(self.dataset_path)

        (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
        x_train = (x_train - 127.5) / 127.5
        # x_train = x_train / 255.0

        y_train = tf.keras.utils.to_categorical(y_train)

        dataset = tf.data.Dataset \
            .from_tensor_slices((x_train, y_train)) \
            .shuffle(self.buffer_size) \
            .batch(self.batch_size)

        tf.data.experimental.save(dataset, self.dataset_path)
        # dataset.save(self.dataset_path)  # 2.9.1

        return dataset

    def _prepare_distribution(self):
        init_sampling = random.normal(0, 1, size=self.buffer_size).astype('float32')
        x_train = init_sampling[(init_sampling < -3) |
                                ((-2 < init_sampling) & (init_sampling < -1)) |
                                ((0 < init_sampling) & (init_sampling < 1)) |
                                ((2 < init_sampling) & (init_sampling < 3))]
        x_train = random.choice(x_train, self.buffer_size)[:, newaxis]

        dataset = tf.data.Dataset \
            .from_tensor_slices(x_train) \
            .batch(self.batch_size)

        tf.data.experimental.save(dataset, self.dataset_path)

        return dataset
