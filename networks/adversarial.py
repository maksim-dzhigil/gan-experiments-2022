import tensorflow as tf

from tensorflow.keras.models import Model

from .abstract_models import AbstractGenerator, AbstractDiscriminator


class Adversarial(Model):
    def __init__(self,
                 generator: AbstractGenerator,
                 discriminator: AbstractDiscriminator,
                 latent_dim: int = 100,
                 batch_size: int = 32,
                 model_name: str = 'adversarial',
                 data_nature: str = 'img_only',
                 num_classes: int = 10
                 ):
        super(Adversarial, self).__init__(name=model_name)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.data_nature = data_nature
        self.num_classes = num_classes

        self.d_optimizer = None
        self.g_optimizer = None
        self.loss_function = None

        self.gen_input_constructor = {
            "img_only": self._make_default_gen_input,
            "img_and_labels": self._make_united_gen_input,
            "distribution": self._make_default_gen_input
        }

        self.dis_input_constructor = {
            "img_only": self._make_default_dis_input,
            "img_and_labels": self._make_united_dis_input,
            "distribution": self._make_default_dis_input
        }

    def compile(self,
                g_optimizer: tf.keras.optimizers = None,
                d_optimizer: tf.keras.optimizers = None,
                loss_function: tf.keras.losses = None
                ):
        super(Adversarial, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_function = loss_function

    @tf.function
    def train_step(self, data):
        # prepare generator input depending on the self.data_nature
        generator_input_dis_training = self.gen_input_constructor[self.data_nature](data)

        generated_images_dis_training = self.generator(generator_input_dis_training)
        # prepare discriminator input depending on the self.data_nature
        dis_input_dis_training = self.dis_input_constructor[self.data_nature](generated_images_dis_training, data)

        labels_for_d_loss = tf.concat([tf.ones((self.batch_size, 1)),
                                       tf.zeros((self.batch_size, 1))], axis=0)
        labels_for_d_loss += 0.05 * tf.random.uniform(tf.shape(labels_for_d_loss))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(dis_input_dis_training, training=True)
            d_loss = self.loss_function(labels_for_d_loss, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        generator_input_gen_training = self.gen_input_constructor[self.data_nature](data)
        generated_images_gen_training = self.generator(generator_input_gen_training)
        dis_input_gen_training = self.dis_input_constructor[self.data_nature](generated_images_gen_training, data)
        misleading_labels = tf.zeros((self.batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(dis_input_gen_training)
            g_loss = self.loss_function(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}

    def _make_default_gen_input(self, data):
        return tf.random.normal(shape=(self.batch_size, self.latent_dim))

    def _make_united_gen_input(self, data):
        _, one_hot_labels = data
        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        return tf.concat([random_latent_vectors, one_hot_labels], axis=1)

    def _make_default_dis_input(self, generated_images, data):
        real_images, _ = data
        return tf.concat([generated_images, real_images], axis=0)

    def _make_united_dis_input(self, generated_images, data):
        real_images, one_hot_labels = data
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[self.generator.image_size * self.generator.image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, self.generator.image_size, self.generator.image_size, self.num_classes)
        )
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)

        return tf.concat([fake_image_and_labels, real_image_and_labels], axis=0)
