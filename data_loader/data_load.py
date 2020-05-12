"""Data loader V.1."""
import pathlib
import json as js
import numpy as np  # pylint: disable=import-error
import tensorflow as tf
import matplotlib.pyplot as plt  # pylint: disable=import-error
from utils.logger import _LOGGER
from tensorflow.keras.utils import to_categorical


class Dataset:  # pylint: disable=too-many-instance-attributes
    """Class for handling datasets."""

    def __init__(self, config):
        """Initialize."""
        _LOGGER.info("Initialize Dataset...")
        self.CONFIG = js.load(config)
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

        self.VAL_NAMES = self.__generate_val_dict()
        self.CLASS_NAMES = self.__generate_class_names()

        self.NAME_DICT = self.__generate_names_dict()

    def __generate_val_dict(self):
        """Generates dict of val images."""
        val_dict = {}

        with open(self.CONFIG["val_names"]) as text_file:

            for line in text_file:
                line_split = line.strip().split("\t")
                val_dict[line_split[0]] = line_split[1].strip()

        return val_dict

    def __generate_names_dict(self):
        """Generate dictionary of names."""
        name_dict = {}

        with open(self.CONFIG["words_path"]) as text_file:

            for line in text_file:
                line_split = line.split("\t")
                words = line_split[1].split(",")
                name_dict[line_split[0]] = words[0]

        return name_dict

    def __generate_class_names(self):
        """Generate the class names."""
        data_dir = pathlib.Path(self.CONFIG["train_path"])
        class_names = np.array([str(item.name) for item in data_dir.glob("*")])

        return class_names

    def __decode_img(self, img_path):
        """Decode JPEG, convert to float [0,1] and resize img."""
        img = tf.image.decode_jpeg(img_path, channels=self.CONFIG["channels"])
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(
            img, [self.CONFIG["image_height"], self.CONFIG["image_width"]]
        )
        return img

    def __parse_data(self, filename, label):
        img_string = tf.io.read_file(filename)
        img = self.__decode_img(img_string)
        return img, label

    def __load_train(self):
        """Loads training dataset."""

        data_dir = pathlib.Path(self.CONFIG["train_path"])

        list_dir = []
        list_label = []

        # Get file paths with corresponding label
        for file_path in list(data_dir.glob("*/images/*.JPEG")):
            file_path = str(file_path)
            file_path_split = file_path.split("/")
            file_path_split = file_path_split[-3]
            index = np.where(self.CLASS_NAMES == file_path_split)
            list_dir.append(file_path)
            list_label.append(index[0])

        try:
            assert self.CONFIG["num_class"] <= 200
        except AssertionError:
            print(
                "The desired number of classes exceeds 200. Changing num_class to 200."
            )
            self.CONFIG["num_class"] = 200

        # Convert label array to numpy array for one hot encoding
        list_label = np.array(
            to_categorical(
                list_label[: self.CONFIG["num_class"] * 500],
                self.CONFIG["num_class"],
                dtype="float32",
            )
        )

        # Convert dir array to numpy array
        list_dir = np.array(list_dir)

        # Convert to tensor labels and dir arrays to tensors
        tf_class = tf.constant(list_label)
        tf_dir = tf.constant(list_dir[: self.CONFIG["num_class"] * 500])

        # Create dataset from tensors and shuffle, map, batch and prefetch
        labeled_ds = tf.data.Dataset.from_tensor_slices((tf_dir, tf_class))
        labeled_ds = labeled_ds.shuffle(buffer_size=len(list_dir))
        labeled_ds = labeled_ds.map(
            self.__parse_data, num_parallel_calls=self.CONFIG["batch_size"]
        )
        labeled_ds = labeled_ds.batch(self.CONFIG["batch_size"])
        labeled_ds = labeled_ds.prefetch(buffer_size=self.CONFIG["batch_size"])
        return labeled_ds

    def __load_val(self):
        """Load valuation data."""
        data_dir = pathlib.Path(self.CONFIG["val_path"])

        list_dir = []
        list_label = []

        # Get file paths with corresponding label
        for file_path in list(data_dir.glob("images/*.JPEG")):
            file_path = str(file_path)
            file_path_split = file_path.split("/")
            file_path_split = file_path_split[-1]
            index = np.where(self.CLASS_NAMES == self.VAL_NAMES[file_path_split])
            list_dir.append(file_path)
            list_label.append(index[0])

        # Convert representation array to numpy array with one hot encoding
        list_label = np.array(to_categorical(list_label, 200, dtype="float32"))

        # Convert dir array to numpy array
        list_dir = np.array(list_dir)

        # Convert to tensor
        tf_class = tf.constant(list_label)
        tf_dir = tf.constant(list_dir)

        # Create, shuffle, map, batch and prefetch dataset
        tf_dir = tf.constant(list_dir)
        tf_class = tf.constant(list_label)
        labeled_ds = tf.data.Dataset.from_tensor_slices((tf_dir, tf_class))
        labeled_ds = labeled_ds.shuffle(buffer_size=len(list_dir))
        labeled_ds = labeled_ds.map(
            self.__parse_data, num_parallel_calls=self.CONFIG["batch_size"]
        )
        labeled_ds = labeled_ds.batch(self.CONFIG["batch_size"])
        labeled_ds = labeled_ds.prefetch(buffer_size=self.CONFIG["batch_size"])

        return labeled_ds

    def get_data(self, dataset="train"):
        """Get images and labels from from dataset."""
        if dataset == "val":
            labeled_ds = self.__load_val()

        elif dataset == "train":
            labeled_ds = self.__load_train()

        return labeled_ds

    def show_batch(self, dataset="train"):
        """Display a random batch."""
        if dataset == "train":
            image_batch, label_batch = next(iter(self.__load_train()))
        else:
            image_batch, label_batch = next(iter(self.__load_val()))

        plt.figure(figsize=(20, 20))
        for index in range(25):
            axis = plt.subplot(5, 5, index + 1)  # pylint: disable=unused-variable
            plt.imshow(image_batch[index])
            plt.title(
                self.NAME_DICT[
                    self.CLASS_NAMES[np.where(label_batch[index].numpy() == 1.0)][0]
                ]
            )  # pylint: disable=singleton-comparison
            plt.axis("off")
