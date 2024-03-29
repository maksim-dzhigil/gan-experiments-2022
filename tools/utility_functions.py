from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MSE
from tensorflow.python.keras.backend import mean


def named_logs(model, logs):
    result = {}
    for log in zip(model.metrics_names, logs):
        result[log[0]] = log[1]
    return result


def binary_cross_entropy(labels, output):
    return BinaryCrossentropy(from_logits=True)(labels, output)


def mean_squared_error(labels, output):
    return MSE(labels, output)


def wasserstein_loss(label, pred):
    return mean(label * pred)
