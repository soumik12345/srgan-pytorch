import torch
from tqdm import tqdm
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
        self.results = {
            'discriminator_loss': [],
            'generator_loss': [],
            'discriminator_score': [],
            'generator_score': [],
            'psnr': [], 'ssim': []
        }

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

    def train_discriminator(self, real_image, fake_image):
        """maximize D(x)-1-D(G(z))"""
        self.discriminator.zero_grad()

        real_output = self.discriminator(real_image).mean()
        fake_output = self.discriminator(fake_image).mean()

        discriminator_loss = 1 - real_output + fake_output
        discriminator_loss.backward(retain_graph=True)
        self.discriminator_optimizer.step()

        return real_output, fake_output, discriminator_loss

    def train_generator(self, z, real_image, fake_image, fake_output):
        """minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss"""
        self.generator.zero_grad()
        generator_loss = self.generator_criterion(fake_output, fake_image, real_image)

        generator_loss.backward()

        fake_image = self.generator(z)
        fake_output = self.discriminator(fake_image).mean()

        self.generator_optimizer.step()

        return generator_loss

    def train_step(self):
        for data, target in tqdm(self.train_dataset):
            batch_size = data.size(0)

            real_image = torch.autograd.Variable(target).cuda()
            z = torch.autograd.Variable(data).cuda()
            fake_image = self.generator(z)

            real_output, fake_output, d_loss = self.train_discriminator(real_image, fake_image)
            g_loss = self.train_generator(z, real_image, fake_image, fake_output)
