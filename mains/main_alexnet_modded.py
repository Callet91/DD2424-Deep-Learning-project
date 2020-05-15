"""Main for alexnet modded v.1."""
from data_loader.data_load import Dataset
from models.alexnet_modded_copy import AlexNetModded


def main():
    """Main."""

    # Load configuration file
    config = open("/workspaces/DD2424-project/configs/alexnet_modded_config.json")

    # Create Dataset of tiny-imagenet from config
    dataset = Dataset(config)

    # Get training data and validation data
    ds_train = dataset.get_data("train")
    ds_test = dataset.get_data("val")

    config_alex = open("/workspaces/DD2424-project/configs/alexnet_modded_config.json")

    # Train pure alexnet according to configuration file
    alex = AlexNetModded(config_alex)
    alex.set_train_data(ds_train)
    alex.set_test_data(ds_test)

    alex.generate_model()
    alex.summary()
    alex.start_train()


if __name__ == "__main__":
    main()
