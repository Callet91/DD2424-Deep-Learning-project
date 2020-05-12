"""Model for AlexNet."""
import json as js
from tensorflow.keras import layers, models, optimizers


class AlexNet:
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
        # TODO: Fix model according to alexnet # pylint: disable=fixme
        self.MODEL = models.Sequential()

        # AlexNet:
        self.MODEL.add(
            layers.Conv2D(
                32,
                (3, 3),
                strides=1,
                activation=self.CONFIG["activation"],
                input_shape=(64, 64, 3),
            )
        )
        # self.MODEL.add(layers.LayerNormalization())
        self.MODEL.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))
        # self.MODEL.add(layers.LayerNormalization())
        self.MODEL.add(
            layers.Conv2D(
                32,
                (3, 3),
                strides=1,
                padding="same",
                activation=self.CONFIG["activation"],
            )
        )
        self.MODEL.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))
        self.MODEL.add(
            layers.Conv2D(
                64,
                (3, 3),
                strides=1,
                padding="same",
                activation=self.CONFIG["activation"],
            )
        )
        self.MODEL.add(
            layers.Conv2D(
                64,
                (3, 3),
                strides=1,
                padding="same",
                activation=self.CONFIG["activation"],
            )
        )
        self.MODEL.add(
            layers.Conv2D(
                64,
                (3, 3),
                strides=1,
                padding="same",
                activation=self.CONFIG["activation"],
            )
        )
        self.MODEL.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))
        self.MODEL.add(layers.Flatten())
        self.MODEL.add(layers.Dense(4096, activation=self.CONFIG["activation"]))
        self.MODEL.add(layers.Dropout(0.5))
        self.MODEL.add(layers.Dense(4096, activation=self.CONFIG["activation"]))
        self.MODEL.add(layers.Dropout(0.5))
        self.MODEL.add(layers.Dense(10, activation="softmax"))

    def start_train(self):
        """Compile and train model."""
        self.MODEL.compile(
            optimizer=self.OPTIMIZER,
            loss="categorical_crossentropy",
            metrics=[self.CONFIG["metrics"]],
        )

        history = self.MODEL.fit(x=self.TRAIN_DS, epochs=self.CONFIG["epochs"])
        return history

    def summary(self):
        """Summay of CNN layers."""
        self.MODEL.summary()
