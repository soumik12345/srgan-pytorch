import torch
import wandb
from tqdm import tqdm
from math import log10
from pytorch_ssim import ssim
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

    def train_step(self):
        self.generator.train()
        self.discriminator.train()

        for data, target in tqdm(self.train_dataset):
            batch_size = data.size(0)

            real_image = torch.autograd.Variable(target).cuda()
            z = torch.autograd.Variable(data).cuda()

            # Update Discriminator: maximize D(x) - 1 - D(G(z))

            fake_image = self.generator(z)
            self.discriminator.zero_grad()
            real_output = self.discriminator(real_image).mean()
            fake_output = self.discriminator(fake_image).mean()

            discriminator_loss = 1 - real_output + fake_output
            discriminator_loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()

            # Update Generator: minimize 1 - D(G(z)) + Perception Loss + Image Loss + TV Loss

            self.generator.zero_grad()
            generator_loss = self.generator_criterion(fake_output, fake_image, real_image)
            generator_loss.backward()

            fake_image = self.generator(z)
            fake_output = self.discriminator(fake_image).mean()

            self.discriminator_optimizer.step()

            wandb.log({
                'generator_loss': generator_loss.item() * batch_size,
                'discriminator_loss': discriminator_loss.item() * batch_size,
                'generator_score': fake_output.item() * batch_size,
                'discriminator_score': real_output.item() * batch_size
            })

    def validation_step(self):
        self.generator.eval()

        for val_lr, val_hr_restore, val_hr in tqdm(self.val_dataset):
            batch_size = val_lr.size(0)
            lr = val_lr.cuda()
            hr = val_hr.cuda()
            sr = self.generator(lr)

            mse = ((sr - hr) ** 2).data.mean()
            structural_similarity = ssim(sr, hr).item()
            psnr = 10 * log10((hr.max() ** 2) / (mse / batch_size))

            wandb.log({
                'Mean Squared Error': mse * batch_size,
                'Structural Similarity': structural_similarity * batch_size,
                'Peak Signal Noise Ratio': psnr
            })

    def train(self):
        for epoch in range(1, self.config['epochs'] + 1):
            print('Epoch:', epoch)
            self.train_step()
            self.validation_step()
