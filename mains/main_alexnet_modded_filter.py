"""Main for alexnet modded v.1."""
# import tensorflow as tf
from data_loader.data_load import Dataset
from models.alexnet_modded_filter import AlexNetModded


def main():
    """Main."""

    # gpus = tf.config.experimental.list_physical_devices("GPU")
    # if gpus:
    #     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0],
    #             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=14336)],
    #         )
    #         logical_gpus = tf.config.experimental.list_logical_devices("GPU")
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as error:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(error)

    # Load configuration file
    config = open("/workspaces/DD2424-project/configs/alexnet_modded_config.json")
    print("Config loaded...")

    # Create Dataset of tiny-imagenet from config
    dataset = Dataset(config)
    print("Dataset created...")

    # Get training data and validation data
    ds_train = dataset.get_data("train")
    ds_test = dataset.get_data("val")

    print("Data extracted...")
    config_alex = open("/workspaces/DD2424-project/configs/alexnet_modded_config.json")

    # Train pure alexnet according to configuration file
    alex = AlexNetModded(config_alex)
    print("Model created...")

    alex.set_train_data(ds_train)
    alex.set_test_data(ds_test)
    print("Data is set to model...")

    alex.generate_model()
    print("Model generated...")

    alex.summary()
    alex.start_train()


if __name__ == "__main__":
    main()
