"""Main app for AlexNet With CIFAR10."""

# Example of imports
from data_loader.data_load import Dataset

# from trainers.example_trainer import ExampleTrainer
# from utils.config import process_config
# from utils.dirs import create_dirs
# from utils.logger import Logger
# from utils.utils import get_args
from models.alexnet import AlexNet  # pylint: disable=import-error


def main():
    """Main function."""

    file = open("/workspaces/DD2424-project/configs/tiny-imagenet.json")
    dataset = Dataset(file)
    (train_images, train_labels) = dataset.get_data(dataset="train")
    (val_images, val_labels) = dataset.get_data(dataset="val")
    dataset.show_batch(dataset="train")

    file = open("/workspaces/DD2424-project/configs/example.json")
    alex = AlexNet(file)

    alex.set_train_data(train_images, train_labels)
    alex.set_test_data(val_images, val_labels)
    # pylint: disable=duplicate-code
    alex.generate_model()
    alex.summary()
    alex.start_train()


if __name__ == "__main__":
    main()
