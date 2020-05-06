"""Model for AlexNet."""
import json as js
from time import time
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.python.keras.callbacks import TensorBoard


class AlexNet:
    """Class for AlexNet."""

    def __init__(self, config):
        """Init Alexnet object."""
        self.model = models.Sequential()
        self.config = js.load(config)
        self.train_imgs = None
        self.train_labels = None
        self.test_imgs = None
        self.test_labels = None

    def set_train_data(self, train_images=None, train_labels=None):
        """Set training data."""
        if (train_images is None) or (train_labels is None):
            self.train_imgs = None
            self.train_labels = None
        else:
            self.train_imgs = train_images
            self.train_labels = train_labels

    def set_test_data(self, test_images=None, test_labels=None):
        """Set test data."""
        if (test_images is None) or (test_labels is None):
            self.test_imgs = None
            self.test_labels = None
        else:
            self.test_imgs = test_images
            self.test_labels = test_labels

    def generate_model(self):
        """Generate model according to Alexnet."""
        # TODO: Fix model according to alexnet # pylint: disable=fixme
        # self.model = models.Sequential()
        self.model.add(
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3))
        )
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation="relu"))
        self.model.add(layers.Dense(10))

    def start_train(self):
        """Compile and train model."""
        self.model.compile(
            optimizer=self.config["optimizer"],
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[self.config["metrics"]],
        )

        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        train = self.model.fit(
            self.train_imgs,
            self.train_labels,
            epochs=self.config["epochs"],
            validation_data=(self.test_imgs, self.test_labels),
            callbacks=[tensorboard],
        )
        return train

    def summary(self):
        """Summay of CNN layers."""
        self.model.summary()
