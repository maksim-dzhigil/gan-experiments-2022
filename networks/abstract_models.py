from tensorflow.keras.models import Model


class AbstractDiscriminator(Model):
    def __init__(self, *args, **kwargs):
        super(AbstractDiscriminator, self).__init__(*args, **kwargs)


class AbstractGenerator(Model):
    def __init__(self, *args, **kwargs):
        super(AbstractGenerator, self).__init__(*args, **kwargs)
