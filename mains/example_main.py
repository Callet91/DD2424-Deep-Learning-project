"""Main app for AlexNet With CIFAR10."""

from data_loader.data_load import Dataset
from models.alexnet import AlexNet  # pylint: disable=import-error


def main():
    """Main function."""
    config = open("/workspaces/DD2424-project/configs/alexnet_config.json")

    dataset = Dataset(config)
    ds_train = dataset.get_data("train")

    config_alex = open("/workspaces/DD2424-project/configs/alexnet_config.json")

    alex = AlexNet(config_alex)
    alex.set_train_data(ds_train)

    alex.generate_model()
    alex.summary()
    alex.start_train()


if __name__ == "__main__":
    main()
