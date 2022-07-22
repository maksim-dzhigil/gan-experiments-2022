import unittest

from kosta_oop import GeneratorDense, DiscriminatorDense, AdversarialNetwork, DataGeneratorForDenseGAN, DCGANTraining, \
    ImageDiscriminatorBasic, ImageGeneratorBasic, DataGeneratorForImageGAN
from tensorflow.keras.models import Model
import tensorflow as tf

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_GeneratorDense(self):
        model = GeneratorDense()
        model.build(input_shape=(None, 1))
        model.call(tf.keras.layers.Input(shape=(1,)))
        model.summary()
        self.assertIsInstance(model, GeneratorDense)

    def test_DiscriminatorDense(self):
        model = DiscriminatorDense()
        model.build(input_shape=(None, 1))
        model.call(tf.keras.layers.Input(shape=(1,)))
        model.summary()
        self.assertIsInstance(model, DiscriminatorDense)

    def test_AdversarialNetwork(self):
        d = DiscriminatorDense()
        g = GeneratorDense()
        model = AdversarialNetwork(d,g)
        model.build(input_shape=(None, 1))
        model.call(tf.keras.layers.Input(shape=(1,)))
        model.summary()
        self.assertIsInstance(model, AdversarialNetwork)

    def test_DataGenerator(self):
        dg = DataGeneratorForDenseGAN()
        X,Y = dg[0] #next(iter(dg))
        self.assertEqual(X.shape, (100, 1))
        self.assertEqual(Y.shape, (100, 1))

    def test_DCGANTraining_Instances(self):
        name = 'test'
        dcgan = DCGANTraining(name=name, gen_vec_size=(None, 1))

        generator = GeneratorDense(name=name + "_gen", output_shape_=dcgan.gen_vec_size[1])
        discriminator = DiscriminatorDense(name=name + "_dis", )
        dcgan.build_models(generator, discriminator)

        self.assertIsInstance(dcgan.discriminator, DiscriminatorDense)
        self.assertIsInstance(dcgan.generator, GeneratorDense)
        self.assertIsInstance(dcgan.adversarial, AdversarialNetwork)


    def test_DCGANTraining_basic(self):

        name = 'test_basic'
        dcgan = DCGANTraining(name=name,
                              latent_size=(None, 100),
                              gen_vec_size=(None, 28, 28, 1))

        dis = ImageDiscriminatorBasic(name=name + "_dis")
        gen = ImageGeneratorBasic(name=name + "_gen")
        dcgan.build_models(gen, dis)

        train_datagen_gen = DataGeneratorForImageGAN(batch_size=100, latent_space_size=100)
        train_datagen_adv = DataGeneratorForImageGAN(batch_size=100, latent_space_size=100, noise_only=True)
        test_gen = DataGeneratorForDenseGAN(batch_size=16, latent_space_size=100)
        dcgan.train_models(train_datagen_gen, train_datagen_adv, test_gen, train_steps=2, save_interval=1)

    def test_DCGANTraining_dense(self):

        name = 'test_dense'
        dcgan = DCGANTraining(name=name, gen_vec_size=(None,1))

        generator = GeneratorDense(name=name + "_gen", output_shape_=dcgan.gen_vec_size[1])
        discriminator = DiscriminatorDense(name=name + "_dis", )
        dcgan.build_models(generator, discriminator)

        train_datagen_gen = DataGeneratorForDenseGAN()
        train_datagen_adv = DataGeneratorForDenseGAN(noise_only=True)
        test_gen = DataGeneratorForDenseGAN()
        dcgan.train_models(train_datagen_gen, train_datagen_adv, test_gen, train_steps=2, save_interval=1)


    #def test_DCGAN_types_of_data(self):
    #    dcgan = DCGAN()
    #    self.assertIsInstance(dcgan.discriminator is DiscriminatorDense)
    #    self.assertIsInstance(dcgan.generator is GeneratorDense)
    #    self.assertIsInstance(dcgan.adversarial is Model)



if __name__ == '__main__':
    unittest.main()
