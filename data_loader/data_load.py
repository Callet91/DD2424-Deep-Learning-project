"""Data loader V.1."""
import pathlib
import json as js
import numpy as np  # pylint: disable=import-error
import tensorflow as tf

# TODO: Uncomment when fix show_batch # pylint: disable=fixme
# import matplotlib.pyplot as plt  # pylint: disable=import-error
from utils.logger import _LOGGER


class Dataset:  # pylint: disable=too-many-instance-attributes
    """Class for handling datasets."""

    def __init__(self, config):
        """Initialize."""
        _LOGGER.info("Initialize Dataset...")
        self.CONFIG = js.load(config)
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

        self.VAL_NAMES = js.load(open(self.CONFIG["val_names_json"]))
        self.CLASS_NAMES = None

        self.TRAIN_PATH = self.CONFIG["train_path"]
        self.VAL_PATH = self.CONFIG["val_path"]
        self.TEST_PATH = self.CONFIG["test_path"]
        self.VALID_DATASETS = ["train", "val", "test"]

        self.IMAGE_HEIGHT = self.CONFIG["image_height"]
        self.IMAGE_WIDTH = self.CONFIG["image_width"]
        self.BATCH_SIZE = self.CONFIG["batch_size"]

        self.TRAIN_IMG_BATCH, self.TRAIN_LABEL_BATCH = self.__load_train()
        self.VAL_IMG_BATCH, self.VAL_LABEL_BATCH = self.__load_val()

    def __decode_img(self, img_path):
        """Decode JPEG, convert to float [0.1] and resize img to 224x224."""
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [self.IMAGE_HEIGHT, self.IMAGE_WIDTH])

    def __get_train_label(self, img_path):
        """Get image label."""
        _LOGGER.info("In __get_train_label: img_path = %s", str(img_path))
        img_label = tf.strings.split(img_path, "/")
        img_label = img_label[-3] == self.CLASS_NAMES
        img_label = tf.where(img_label)
        return img_label

    def __process_train_img(self, img_path):
        """Get label and decode image."""
        _LOGGER.info("In __process__img: Path = %s", str(img_path))
        label = self.__get_train_label(img_path)
        img = self.__decode_img(img_path)
        return img, label

    def __prepare_train_data(self, dataset, cache=True, shuffle_buffer_size=1000):
        """Cache and shuffle dataset."""
        if cache:
            if isinstance(cache, str):
                dataset.cache(cache)
            else:
                dataset.cache()

        dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=self.AUTOTUNE)

        return dataset

    def __load_train(self):
        """Loads training dataset."""

        try:
            assert self.TRAIN_PATH is not None
        except AssertionError as error:
            raise error

        # Check path
        path_split = self.TRAIN_PATH.split("/")

        try:
            assert path_split[-2] in self.VALID_DATASETS
        except AssertionError as error:
            raise error

        data_dir = pathlib.Path(self.TRAIN_PATH)
        list_ds = tf.data.Dataset.list_files(str(data_dir / "*/images/*.JPEG"))

        self.CLASS_NAMES = np.array([item.name for item in data_dir.glob("*")])

        # Process all data and map image with label
        labeled_ds = list_ds.map(
            self.__process_train_img, num_parallel_calls=self.AUTOTUNE
        )

        # Prepare data for training
        dataset = self.__prepare_train_data(labeled_ds)

        img_batch, label_batch = next(iter(dataset))

        return img_batch, label_batch

    def __get_label_val(self, img_label):
        """Get image label."""

        return img_label == self.CLASS_NAMES

    def __load_val(self):

        data_dir = pathlib.Path(self.VAL_PATH)

        val_dirs = []
        val_class = []

        for file_path in list(data_dir.glob("images/*.JPEG")):
            file_path = str(file_path)
            file_path_split = file_path.split("/")
            file_path_split = file_path_split[-1]
            val_dirs.append(file_path)
            index = np.where(self.CLASS_NAMES == self.VAL_NAMES[file_path_split])
            val_class.append(index)

        list_dir = tf.data.Dataset.from_tensor_slices(val_dirs)
        label = tf.data.Dataset.from_tensor_slices(val_class)

        label = label.batch(self.BATCH_SIZE)
        label = label.prefetch(buffer_size=self.AUTOTUNE)
        label = next(iter(label))

        img = list_dir.map(self.__decode_img)
        img = img.batch(self.BATCH_SIZE)
        img = img.prefetch(buffer_size=self.AUTOTUNE)
        img = next(iter(img))

        return img, label

    def get_data(self, dataset="train"):
        """Get images and labels from from dataset."""
        if dataset == "val":
            img, label = self.__load_val()

        elif dataset == "train":
            img, label = self.__load_train()

        return img, label

    # TODO: Fix one hot rep for val and train batch for printing # pylint: disable=fixme
    def show_batch(self, dataset="train"):
        """Display a 5x5 grid of random img from dataset."""
        if dataset == "train":
            img = self.TRAIN_IMG_BATCH
            label = self.TRAIN_LABEL_BATCH

        elif dataset == "val":
            img = self.VAL_IMG_BATCH
            label = self.VAL_LABEL_BATCH

        print(img)
        print(label)

        # plt.figure(figsize=(10, 10))
        # for n_img in range(25):
        #     axis = plt.subplot(5, 5, n_img + 1)
        #     _LOGGER.info(axis)
        #     plt.imshow(img[n_img])
        #     title = self.CLASS_NAMES[if label[n_img] is True][
        #         0
        #     ].title()  # pylint: disable=singleton-comparison
        #     plt.title(title)
        #     plt.axis("off")
        # plt.show()
