import os
import torch
import wandb
import random
import torchvision
from glob import glob
from tqdm import tqdm
from math import log10
from src.ssim import ssim
from secret import WANDB_API_KEY
from src.loss import GeneratorLoss
from src.models import Generator, Discriminator
from src.dataset import TrainDataset, ValidationDataset


class Trainer:

    def __init__(self, config):
        self.config = config
        self.device = self.config['device']
        self.initialize_wandb()
        self.visualization_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(400),
            torchvision.transforms.CenterCrop(400)
        ])
        self.train_dataset, self.val_dataset = self.get_dataloaders()
        self.generator, self.discriminator = self.get_models()
        self.generator_criterion = GeneratorLoss().to(self.device)
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
        generator = Generator(self.config['scale']).to(self.device)
        discriminator = Discriminator().to(self.device)
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

            real_image = target.to(self.device)
            z = data.to(self.device)

            # Update Discriminator: maximize D(x) - 1 - D(G(z))

            self.discriminator_optimizer.zero_grad()
            fake_image = self.generator(z)
            real_output = self.discriminator(real_image).mean()
            fake_output = self.discriminator(fake_image).mean()

            discriminator_loss = 1 - real_output + fake_output
            discriminator_loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()

            # Update Generator: minimize 1 - D(G(z)) + Perception Loss + Image Loss + TV Loss

            self.generator_optimizer.zero_grad()
            generator_loss = self.generator_criterion(fake_output, fake_image, real_image)
            generator_loss.backward()
            self.discriminator_optimizer.step()

            wandb.log({
                'generator_loss': generator_loss.item() * batch_size,
                'discriminator_loss': discriminator_loss.item() * batch_size,
                'generator_score': fake_output.item() * batch_size,
                'discriminator_score': real_output.item() * batch_size
            })

    def validation_step(self):
        self.generator.eval()
        with torch.no_grad():
            for i in tqdm(range(20)):
                for val_lr, val_hr, val_hr_restore in self.val_dataset:
                    batch_size = val_lr.size(0)
                    lr = val_lr.to(self.device)
                    hr = val_hr.to(self.device)
                    sr = self.generator(lr)

                    mse = ((sr - hr) ** 2).data.mean()
                    structural_similarity = ssim(sr, hr).item()
                    psnr = 10 * log10((hr.max() ** 2) / (mse / batch_size))

                    wandb.log({
                        'Mean Squared Error': mse * batch_size,
                        'Structural Similarity': structural_similarity * batch_size,
                        'Peak Signal Noise Ratio': psnr,
                    })

                    if i == 0:
                        wandb.log({
                            "Validation Images": [
                                wandb.Image(lr.data.cpu().squeeze(0), caption="Low-Res"),
                                wandb.Image(hr.data.cpu().squeeze(0), caption="High-Res"),
                                wandb.Image(sr.data.cpu().squeeze(0), caption="Super-Res")
                            ]
                        })

    def train(self):
        wandb.watch(self.generator)
        for epoch in range(1, self.config['epochs'] + 1):
            print('Epoch:', epoch)
            self.train_step()
            self.validation_step()


if __name__ == '__main__':
    # remove 2008_001823.jpg
    images = glob('./data/VOCdevkit/VOC2012/JPEGImages/*')
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
        'val_batch_size': 1,
        'epochs': 100,
        'device': "cuda:0"
    }

    trainer = Trainer(configurations)
    trainer.train()
