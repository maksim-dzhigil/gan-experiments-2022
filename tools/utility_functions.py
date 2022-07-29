from tensorflow.keras.losses import BinaryCrossentropy


def named_logs(model, logs):
    result = {}
    for log in zip(model.metrics_names, logs):
        result[log[0]] = log[1]
    return result


def loss_fn(labels, output):
    return BinaryCrossentropy(from_logits=True)(labels, output)
