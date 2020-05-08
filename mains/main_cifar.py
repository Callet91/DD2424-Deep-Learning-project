"""Main app for AlexNet With CIFAR10."""
from tensorflow.keras import datasets

# Example of imports
# from data_loader.data_generator import DataGenerator
# from models.example_model import ExampleModel
# from trainers.example_trainer import ExampleTrainer
# from utils.config import process_config
# from utils.dirs import create_dirs
# from utils.logger import Logger
# from utils.utils import get_args
from models.alexnet import AlexNet  # pylint: disable=import-error


def main():
    """Main function."""
    file = open("/workspaces/DD2424-project/configs/example.json")
    alex = AlexNet(file)

    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = datasets.cifar10.load_data()
    train_images, test_images = (
        train_images / 255.0,
        test_images / 255.0,
    )  # Normalize pixel values to be between 0 and 1
    alex.set_train_data(train_images, train_labels)
    alex.set_test_data(test_images, test_labels)

    # alex.generate_model()
    # alex.summary()
    # alex.start_train()


if __name__ == "__main__":
    main()
