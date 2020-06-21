from src.dataset import TrainDataset, ValidationDataset, TestDataset


class Trainer:

    def __init__(self, config):
        self.config = config
        self.train_dataset, self.val_dataset = self.get_dataloaders()

    def get_dataloaders(self):
        train_dataset = TrainDataset(
            self.config['train_images'],
            self.config['crop_size'], self.config['scale']
        ).get_loader(
            self.config['num_workers'],
            self.config['batch_size']
        )
        val_dataset = ValidationDataset(
            self.config['val_images'],
            self.config['scale']
        ).get_loader(
            self.config['num_workers'],
            self.config['batch_size']
        )
        return train_dataset, val_dataset
