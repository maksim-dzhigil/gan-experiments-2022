import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model

from .abstract_models import AbstractGenerator, AbstractDiscriminator


class ConditionalAdversarialNetwork(Model):
    def __init__(self,
                 generator: AbstractGenerator,
                 discriminator: AbstractDiscriminator,
                 latent_dim: int = 100,
                 batch_size: int = 32,
                 model_name: str = 'conditional_adversarial_network',
                 num_classes: int = 10
                 ):
        super(ConditionalAdversarialNetwork, self).__init__(name=model_name)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.d_optimizer = None
        self.g_optimizer = None
        self.loss_fn = None

    def compile(self,
                g_optimizer: tf.keras.optimizers = None,
                d_optimizer: tf.keras.optimizers = None,
                loss_function: tf.keras.losses = None
                ):
        super(ConditionalAdversarialNetwork, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_function

    @tf.function
    def train_step(self, data):
        real_images, one_hot_labels = data

        random_latent_vectors = tf.random.uniform(shape=(self.batch_size, self.latent_dim))
        generator_input = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )
        generated_images = self.generator(generator_input)

        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[self.generator.image_size * self.generator.image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, self.generator.image_size, self.generator.image_size, self.num_classes)
        )

        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        discriminator_input = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        labels_for_d_loss = tf.concat(
            [tf.ones((self.batch_size, 1)), tf.zeros((self.batch_size, 1))], axis=0
        )

        labels_for_d_loss += 0.05 * tf.random.uniform(tf.shape(labels_for_d_loss))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(discriminator_input)
            d_loss = self.loss_fn(labels_for_d_loss, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        generator_input = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        misleading_labels = tf.zeros((self.batch_size, 1))

        with tf.GradientTape() as tape:
            fake_images = self.generator(generator_input)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}


class AdversarialNetwork(Model):

    def __init__(self,
                 generator: AbstractGenerator,
                 discriminator: AbstractDiscriminator,
                 latent_dim: int = 100,
                 batch_size: int = 32,  # 32
                 model_name: str = 'adversarial_network',
                 data_nature: str = 'mnist'  # one_dim
                 ):
        super(AdversarialNetwork, self).__init__(name=model_name)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.data_nature = data_nature

        self.d_optimizer = None
        self.g_optimizer = None
        self.loss_fn = None

    def compile(self,
                g_optimizer: tf.keras.optimizers = None,
                d_optimizer: tf.keras.optimizers = None,
                loss_function: tf.keras.losses = None
                ):
        super(AdversarialNetwork, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_function

    @tf.function
    def train_step(self, data):
        if self.data_nature == 'one_dim':
            real_images = data        # for one_dim
        if self.data_nature == 'mnist':
            real_images, _ = data   # for mnist
        generator_input = tf.random.uniform(shape=(self.batch_size, self.latent_dim))
        generated_images = self.generator(generator_input)

        discriminator_input = tf.concat([generated_images, real_images], axis=0)

        labels_for_d_loss = tf.concat([tf.ones((self.batch_size, 1)),
                                       tf.zeros((self.batch_size, 1))], axis=0)
        labels_for_d_loss += 0.05 * tf.random.uniform(tf.shape(labels_for_d_loss))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(discriminator_input, training=True)
            d_loss = self.loss_fn(labels_for_d_loss, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        generator_input = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        misleading_labels = tf.zeros((self.batch_size, 1))

        with tf.GradientTape() as tape:
            discriminator_input = self.generator(generator_input)
            predictions = self.discriminator(discriminator_input)
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}


class WassersteinAdversarial(Model):
    def __init__(self,
                 generator: AbstractGenerator,
                 discriminator: AbstractDiscriminator,
                 latent_dim: int = 100,
                 batch_size: int = 32,
                 model_name: str = 'conditional_adversarial_network',
                 num_classes: int = 10,
                 n_critic: int = 5,
                 clip_value: float = 0.01
                 ):
        super(WassersteinAdversarial, self).__init__(name=model_name)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.n_critic = n_critic
        self.clip_value = clip_value

        self.d_optimizer = None
        self.g_optimizer = None
        self.loss_fn = None

    def compile(self,
                g_optimizer: tf.keras.optimizers = None,
                d_optimizer: tf.keras.optimizers = None,
                loss_function: tf.keras.losses = None
                ):
        super(WassersteinAdversarial, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_function

    @tf.function
    def train_step(self, data):
        real_images, _ = data
        d_loss = 0
        for _ in range(self.n_critic):
            generator_input = tf.random.uniform(shape=(self.batch_size, self.latent_dim))
            generated_images = self.generator(generator_input)

            real_labels = tf.ones((self.batch_size, 1))
            fake_labels = -(tf.ones((self.batch_size, 1)))

            real_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))
            fake_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))

            with tf.GradientTape() as tape:
                fake_pred = self.discriminator(generated_images)
                real_pred = self.discriminator(real_images)
                fake_loss = self.loss_fn(fake_labels, fake_pred)
                real_loss = self.loss_fn(real_labels, real_pred)
                uni_loss = 0.5 * (real_loss + fake_loss)

            d_loss += uni_loss
            grads = tape.gradient(uni_loss, self.discriminator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

            for layer in self.discriminator.layers:
                if len(layer.trainable_weights) != 0:
                    weights = layer.trainable_weights  # .get_weights()
                    # weights = [tf.clip_by_value(weights,
                    #                    -self.clip_value,
                    #                    self.clip_value)]
                    layer.set_weights(weights)

        d_loss /= self.n_critic

        generator_input = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        misleading_labels = tf.zeros((self.batch_size, 1))

        with tf.GradientTape() as tape:
            discriminator_input = self.generator(generator_input)
            predictions = self.discriminator(discriminator_input)
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}





