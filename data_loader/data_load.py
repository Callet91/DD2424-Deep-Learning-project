"""Data loader V.1."""
import pathlib
import tensorflow as tf
from utils.logger import _LOGGER

# import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
CLASS_NAMES = None
VALID_DATASETS = ["test", "train", "val"]


def __get_label(img_path):
    """Get image label."""
    img_label = tf.strings.split(img_path, "/")
    # TODO: Fix CLASS NAMES # pylint: disable=fixme
    return img_label[-2] == CLASS_NAMES


def __decode_img(img):
    """Decode jpg, convert to float [0.1] and resize img to 256x256."""
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [224, 224])


def __process_img(img_path):
    """Get label and decode image."""
    label = __get_label(img_path)
    img = tf.io.read_file(img_path)
    img = __decode_img(img)
    return img, label


def load_dataset(path=None, name_dataset=None, n_img=10):
    """Loads  dataset."""

    assert path is not None
    assert name_dataset is not None

    path = path + "/" + name_dataset

    _LOGGER.info("In loadDataset: Path is set to %s", path)
    data_dir = pathlib.Path(path)
    list_ds = tf.data.Dataset.list_files(str(data_dir / "*/images/*.JPEG"))

    for fimg in list_ds.take(n_img):
        print(fimg.numpy())

    labeled_ds = list_ds.map(__process_img, num_parallel_calls=AUTOTUNE)

    for image, label in labeled_ds.take(n_img):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    print(CLASS_NAMES)

    return 0
