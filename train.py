import os
import torch
import wandb
import random
import torchvision
from glob import glob
from tqdm import tqdm
from math import log10
from pytorch_ssim import ssim
from secret import WANDB_API_KEY
from src.loss import GeneratorLoss
from src.models import Generator, Discriminator
from src.dataset import TrainDataset, ValidationDataset


class Trainer:

    def __init__(self, config):
        self.config = config
        self.initialize_wandb()
        self.visualization_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(400),
            torchvision.transforms.CenterCrop(400)
        ])
        self.train_dataset, self.val_dataset = self.get_dataloaders()
        self.generator, self.discriminator = self.get_models()
        self.generator_criterion = GeneratorLoss().cuda()
        self.generator_optimizer, self.discriminator_optimizer = self.get_optimizers()

    def initialize_wandb(self):
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        wandb.init(
            project=self.config['project_name'],
            name=self.config['experiment_name']
        )

    def get_dataloaders(self):
        train_dataset = TrainDataset(
            self.config['train_images'],
            self.config['crop_size'], self.config['scale']
        ).get_loader(
            self.config['num_workers'],
            self.config['train_batch_size']
        )
        val_dataset = ValidationDataset(
            self.config['val_images'],
            self.config['scale']
        ).get_loader(
            self.config['num_workers'],
            self.config['val_batch_size']
        )
        return train_dataset, val_dataset

    def get_models(self):
        generator = Generator(self.config['scale']).cuda()
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

        iteration = 0
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
                'Peak Signal Noise Ratio': psnr,
            })

            if iteration == 0:
                wandb.log({
                    "Validation Images": [
                        wandb.Image(lr.data.cpu().squeeze(0), caption="Low-Res"),
                        wandb.Image(hr.data.cpu().squeeze(0), caption="High-Res"),
                        wandb.Image(sr.data.cpu().squeeze(0), caption="Super-Res")
                    ]
                })

            iteration += 1

    def train(self):
        wandb.watch(self.generator)
        for epoch in range(1, self.config['epochs'] + 1):
            print('Epoch:', epoch)
            self.train_step()
            self.validation_step()


if __name__ == '__main__':
    images = glob('./data/VOC2012/JPEGImages/*')
    random.shuffle(images)
    train_images = images[:17000]
    val_images = images[17000:]

    configurations = {
        'project_name': 'srgan-pytorch',
        'experiment_name': 'exp-1',
        'train_images': train_images,
        'val_images': val_images,
        'crop_size': 88,
        'scale': 2,
        'num_workers': 4,
        'train_batch_size': 8,
        'val_batch_size': 4,
        'epochs': 100
    }

    trainer = Trainer(configurations)
    trainer.train()
