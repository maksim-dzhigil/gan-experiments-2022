import tensorflow as tf

from tensorflow.keras.models import Model

from abstract_models import AbstractGenerator, AbstractDiscriminator


class ConditionalAdversarialNetwork(Model):
    def __init__(self,
                 generator: AbstractGenerator,
                 discriminator: AbstractDiscriminator,
                 latent_dim: int = 100,
                 batch_size: int = 32,
                 model_name: str = 'adversarial_network',
                 num_classes: int = 10  # for conditional gan
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

        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[self.generator.image_size * self.generator.image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, self.generator.image_size, self.generator.image_size, self.num_classes)
        )

        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        generated_images = self.generator(random_vector_labels)

        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        labels = tf.concat(
            [tf.ones((self.batch_size, 1)), tf.zeros((self.batch_size, 1))], axis=0
        )

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        misleading_labels = tf.zeros((self.batch_size, 1))

        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"g_loss": g_loss, "d_loss": d_loss}


class AdversarialNetwork(Model):

    def __init__(self,
                 generator: AbstractGenerator,
                 discriminator: AbstractDiscriminator,
                 latent_dim: int = 100,
                 batch_size: int = 100,  # 32
                 model_name: str = 'adversarial_network',
                 ):
        super(AdversarialNetwork, self).__init__(name=model_name)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.batch_size = batch_size

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
        real_images = data        # for one_dim
        # real_images, _ = data   # for mnist
        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        generated_images = self.generator(random_latent_vectors)
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat([tf.ones((self.batch_size, 1)),
                            tf.zeros((self.batch_size, 1))], axis=0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        misleading_labels = tf.zeros((self.batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}
