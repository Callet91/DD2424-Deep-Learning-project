"""Main app for AlexNet With CIFAR10."""
from data_loader.data_load import Dataset

# Example of imports
# from data_loader.data_generator import DataGenerator
# from models.example_model import ExampleModel
# from trainers.example_trainer import ExampleTrainer
# from utils.config import process_config
# from utils.dirs import create_dirs
# from utils.logger import Logger
# from utils.utils import get_args
from models.cifar_model import AlexNet  # pylint: disable=import-error


def main():
    """Main function."""
    config = open("/workspaces/DD2424-project/configs/tiny-imagenet.json")

    dataset = Dataset(config)
    ds_train = dataset.get_data("train")

    config_alex = open("/workspaces/DD2424-project/configs/example.json")

    alex = AlexNet(config_alex)
    alex.set_train_data(ds_train)

    alex.generate_model()
    alex.summary()
    alex.start_train()


if __name__ == "__main__":
    main()
