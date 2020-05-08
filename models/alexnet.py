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
        self.MODEL = models.Sequential()
        self.CONFIG = js.load(config)
        self.TRAIN_IMGS = None
        self.TRAIN_LABELS = None
        self.TEST_IMGS = None
        self.TEST_LABELS = None

    def set_train_data(self, train_images=None, train_labels=None):
        """Set training data."""
        if (train_images is None) or (train_labels is None):
            self.TRAIN_IMGS = None
            self.TRAIN_LABELS = None
        else:
            self.TRAIN_IMGS = train_images
            self.TRAIN_LABELS = train_labels

    def set_test_data(self, test_images=None, test_labels=None):
        """Set test data."""
        if (test_images is None) or (test_labels is None):
            self.TEST_IMGS = None
            self.TEST_LABELS = None
        else:
            self.TEST_IMGS = test_images
            self.TEST_LABELS = test_labels

    def generate_model(self):
        """Generate model according to Alexnet."""
        # TODO: Fix model according to alexnet # pylint: disable=fixme
        # self.MODEL = models.Sequential()
        # trial CIFAR10:
        # self.MODEL.add(
        #     layers.Conv2D(32, (3, 3), activation=self.CONFIG["activation"],
        # input_shape=(32, 32, 3))
        # )
        # self.MODEL.add(layers.MaxPooling2D((2, 2)))
        # self.MODEL.add(layers.Conv2D(64, (3, 3), activation=self.CONFIG["activation"]))
        # self.MODEL.add(layers.MaxPooling2D((2, 2)))
        # self.MODEL.add(layers.Conv2D(64, (3, 3), activation=self.CONFIG["activation"]))
        # self.MODEL.add(layers.Flatten())
        # self.MODEL.add(layers.Dense(64, activation=self.CONFIG["activation"]))
        # self.MODEL.add(layers.Dense(10))

        # AlexNet:
        self.MODEL.add(
            layers.Conv2D(
                96,
                (11, 11),
                strides=(4, 4),
                activation=self.CONFIG["activation"],
                input_shape=(227, 227, 3),
            )
        )
        self.MODEL.add(layers.MaxPooling2D((3, 3), strides=2))
        self.MODEL.add(
            layers.LayerNormalization(
                beta_initializer="zeros", gamma_initializer="ones"
            )
        )
        self.MODEL.add(
            layers.Conv2D(
                256,
                (5, 5),
                strides=1,
                padding="same",
                activation=self.CONFIG["activation"],
            )
        )
        self.MODEL.add(layers.MaxPooling2D((3, 3), strides=2))
        self.MODEL.add(
            layers.LayerNormalization(
                beta_initializer="zeros", gamma_initializer="ones"
            )
        )
        self.MODEL.add(
            layers.Conv2D(
                384,
                (3, 3),
                strides=1,
                padding="same",
                activation=self.CONFIG["activation"],
            )
        )
        self.MODEL.add(
            layers.Conv2D(
                384,
                (3, 3),
                strides=1,
                padding="same",
                activation=self.CONFIG["activation"],
            )
        )
        self.MODEL.add(
            layers.Conv2D(
                256,
                (3, 3),
                strides=1,
                padding="same",
                activation=self.CONFIG["activation"],
            )
        )
        self.MODEL.add(layers.MaxPooling2D((3, 3), strides=2))
        # self.MODEL.add(layers.Flatten())
        self.MODEL.add(layers.Dense(4096, activation=self.CONFIG["activation"]))
        # self.MODEL.add(layers.Dropout(0.5))
        self.MODEL.add(layers.Dense(4096, activation=self.CONFIG["activation"]))
        # self.MODEL.add(layers.Dropout(0.5))
        self.MODEL.add(layers.Dense(200, activation="softmax"))

    def start_train(self):
        """Compile and train model."""
        self.MODEL.compile(
            optimizer=self.CONFIG["optimizer"],
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[self.CONFIG["metrics"]],
        )

        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        history = self.MODEL.fit(
            self.TRAIN_IMGS,
            self.TRAIN_LABELS,
            epochs=self.CONFIG["epochs"],
            validation_data=(self.TEST_IMGS, self.TEST_LABELS),
            callbacks=[tensorboard],
        )
        return history

    def summary(self):
        """Summay of CNN layers."""
        self.MODEL.summary()
