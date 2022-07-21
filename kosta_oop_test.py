import unittest

from kosta_oop import GeneratorDense, DiscriminatorDense, AdversarialNetwork, DataGenerator, DCGANTraining
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
        model = AdversarialNetwork()
        model.build(input_shape=(None, 1))
        model.call(tf.keras.layers.Input(shape=(1,)))
        model.summary()
        self.assertIsInstance(model, AdversarialNetwork)

    def test_DataGenerator(self):
        dg = DataGenerator()
        X,Y = dg[0] #next(iter(dg))
        self.assertEqual(X.shape, (100, 1))
        self.assertEqual(Y.shape, (100, 1))

    def test_DCGANTraining_Instances(self):
        train = DCGANTraining()
        train.build_models()
        self.assertIsInstance(train.discriminator, DiscriminatorDense)
        self.assertIsInstance(train.generator, GeneratorDense)
        self.assertIsInstance(train.adversarial, AdversarialNetwork)


    #def test_DCGAN_types_of_data(self):
    #    dcgan = DCGAN()
    #    self.assertIsInstance(dcgan.discriminator is DiscriminatorDense)
    #    self.assertIsInstance(dcgan.generator is GeneratorDense)
    #    self.assertIsInstance(dcgan.adversarial is Model)



if __name__ == '__main__':
    unittest.main()
