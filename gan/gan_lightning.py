import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import numpy as np

from lightning import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.strategies import DDPStrategy

import argparse

class MNISTDataset(LightningDataModule):
    def __init__(
            self,
            batch_size: int = 64,
            num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = "./data"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (1, 28, 28)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.hparams.batch_size, num_workser=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.hparams.batch_size, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size, num_workers=self.num_workers)
    
class Generator(nn.Module):
    def __init__(self , latent_dim, img_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh(),
        )
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
    

class GAN(LightningModule):
    def __init__(
             self,
             channels: int,
             width: int,
             height: int,
             latent_dim: int = 100,
             generator_lr: float = 0.0002, 
             discriminator_lr: float = 0.0002,
             b1: float = 0.5,
             b2: float = 0.999,
             batch_size: int = 64,
             **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False


        data_shape = (channels, width, height)
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=data_shape) 
        self.discriminator = Discriminator(img_shape=data_shape)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, real_images, fake_images):
        return F.binary_cross_entropy()
    
    def configure_optimizers(self):
        g_lr = self.hparams.generator_lr
        d_lr = self.hparams.discriminator_lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=g_lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=d_lr, betas=(b1, b2))
        return [opt_g, opt_d], []
    
    def training_step(self, batch, batch_idx):

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GAN')
    parser.add_argument('--device', type=int)
    parser.add_argument('--num_nodes', type=int)
    parser.add_argument('--precision', type=str, default='bf16')
    args = parser.parse_args()

    data_module = MNISTDAtaModule()
    model = GAN(*data_module.dims)

    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
    trainer = Trainer(
        accelerator='gpu',
        devices=args.device,
        num_nodes=args.num_nodes,
        precision=args.precision,
        max_epochs=10,
        strategy='ddp',
    )
    trainer.fit(model, data_module)