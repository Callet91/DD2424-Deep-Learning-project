"""Model for AlexNet modded."""
import os
import datetime
import json as js
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


class AlexNetModded:
    """Class for AlexNet."""

    def __init__(self, config):
        """Init Alexnet object."""
        self.MODEL = models.Sequential()
        self.CONFIG = js.load(config)
        self.TRAIN_DS = None
        self.TEST_DS = None
        self.OPTIMIZER = optimizers.SGD(
            lr=self.CONFIG["learning_rate"],
            decay=self.CONFIG["decay"],
            momentum=self.CONFIG["momentum"],
        )

    def set_train_data(self, train_ds=None):
        """Set training data."""
        self.TRAIN_DS = train_ds

    def set_test_data(self, test_ds=None):
        """Set test data."""
        self.TEST_DS = test_ds

    def generate_model(self):
        """Generate model according to Alexnet."""
        self.MODEL = models.Sequential()

        # AlexNet:
        self.MODEL.add(
            layers.Conv2D(
                96,
                (3, 3),
                strides=1,
                activation=self.CONFIG["activation"],
                input_shape=(
                    self.CONFIG["image_height"],
                    self.CONFIG["image_width"],
                    self.CONFIG["channels"],
                ),
            )
        )
        self.MODEL.add(layers.LayerNormalization())
        self.MODEL.add(layers.MaxPooling2D(pool_size=(2, 2), strides=1))
        self.MODEL.add(layers.LayerNormalization())
        self.MODEL.add(
            layers.Conv2D(
                128,
                (3, 3),
                strides=1,
                padding="same",
                activation=self.CONFIG["activation"],
            )
        )
        self.MODEL.add(layers.MaxPooling2D(pool_size=(2, 2), strides=1))
        self.MODEL.add(
            layers.Conv2D(
                128,
                (3, 3),
                strides=1,
                padding="same",
                activation=self.CONFIG["activation"],
            )
        )
        self.MODEL.add(
            layers.Conv2D(
                128,
                (3, 3),
                strides=1,
                padding="same",
                activation=self.CONFIG["activation"],
            )
        )
        self.MODEL.add(
            layers.Conv2D(
                128,
                (3, 3),
                strides=1,
                padding="same",
                activation=self.CONFIG["activation"],
            )
        )
        self.MODEL.add(layers.MaxPooling2D(pool_size=(2, 2), strides=1))
        self.MODEL.add(layers.Flatten())
        self.MODEL.add(layers.Dropout(0.5))
        self.MODEL.add(layers.Dense(4096, activation=self.CONFIG["activation"]))
        self.MODEL.add(layers.Dropout(0.5))
        self.MODEL.add(layers.Dense(4096, activation=self.CONFIG["activation"]))
        self.MODEL.add(layers.Dense(self.CONFIG["num_class"], activation="softmax"))

    def start_train(self):
        """Compile and train model."""

        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        self.MODEL.compile(
            optimizer=self.OPTIMIZER,
            loss="categorical_crossentropy",
            metrics=[self.CONFIG["metrics"]],
        )

        if self.TEST_DS is not None:
            history = self.MODEL.fit(
                x=self.TRAIN_DS,
                epochs=self.CONFIG["epochs"],
                callbacks=[tensorboard_callback],
                validation_data=self.TEST_DS,
                shuffle=True,
            )

        else:
            history = self.MODEL.fit(
                x=self.TRAIN_DS,
                epochs=self.CONFIG["epochs"],
                callbacks=[tensorboard_callback],
                shuffle=True,
            )
        return history

    def summary(self):
        """Summay of CNN layers."""
        self.MODEL.summary()

    def evaluate(self):
        """Evaluate on test dataset."""
        result = self.MODEL.evaluate(self.TEST_DS)
        return result
