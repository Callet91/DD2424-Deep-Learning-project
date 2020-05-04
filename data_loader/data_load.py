import tensorflow as tf
import pathlib

# import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
CLASS_NAMES = None

"""Data loader V.1."""


def __get_list_ds(path_dataset, n_img=100):
    """Get list of paths to images."""
    img_paths = [None] * n_img
    list_ds = tf.data.Dataset.list_files(path_dataset)

    for f in list_ds.take(n_img):
        img_paths.append(f)

    return img_paths


def __get_label(img_path):
    """Get image label."""
    img_label = tf.strings.split(img_path, "/")
    # TODO: Fix CLASS NAMES
    return img_label[-2] == CLASS_NAMES


def __decode_img(img):
    """Decode jpg, convert to float [0.1] and resize img to 256x256."""
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [224, 224])


def __process_img(img_path):
    label = __get_label(img_path)
    img = tf.io.read_file(img_path)
    img = __decode_img(img)
    return img, label


def loadDataset(path, n_img):
    """Loads  dataset  """
    # TODO: Manage data_dir and delete print
    data_dir = pathlib.Path(path)
    print(data_dir)
    path_dataset = path + "/*/images/*"
    # TODO: Manage img_path and delete pir
    img_paths = __get_list_ds(path_dataset, n_img)
    print(img_paths)

    return 1
