import torch
from src.loss import GeneratorLoss
from src.models import Generator, Discriminator
from src.dataset import TrainDataset, ValidationDataset


class Trainer:

    def __init__(self, config):
        self.config = config
        self.train_dataset, self.val_dataset = self.get_dataloaders()
        self.generator, self.discriminator = self.get_models()
        self.generator_criterion = GeneratorLoss().cuda()
        self.generator_optimizer, self.discriminator_optimizer = self.get_optimizers()

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

    def get_models(self):
        generator = Generator(self.config['self']).cuda()
        discriminator = Discriminator().cuda()
        return generator, discriminator

    def get_optimizers(self):
        generator_optimizer = torch.optim.Adam(self.generator.parameters())
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters())
        return generator_optimizer, discriminator_optimizer
