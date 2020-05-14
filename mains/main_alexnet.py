"""Main for alexnet v.1."""
from models.alexnet import AlexNet
from data_loader.data_load import Dataset


def main():
    """Main function."""
    # Load configuration file
    config = open("/workspaces/DD2424-project/configs/alexnet_config.json")

    # Create Dataset of tiny-imagenet from config
    dataset = Dataset(config)

    # Get training data and validation data
    ds_train = dataset.get_data("train")
    ds_test = dataset.get_data("val")

    config_alex = open("/workspaces/DD2424-project/configs/alexnet_config.json")

    # Train pure alexnet according to configuration file
    alex = AlexNet(config_alex)
    alex.set_train_data(ds_train)
    alex.set_test_data(ds_test)

    alex.generate_model()
    alex.summary()
    alex.start_train()


if __name__ == "__main__":
    main()
