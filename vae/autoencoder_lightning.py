from torch import nn, optim, utils
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


import lightning
from lightning import Trainer, LightningModule
from lightning.pytorch.loggers import WandbLogger
import wandb
import argparse

class AutoEncoder(LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.veiw(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AutoEncoder')
    parser.add_argument('--wandb_key', type=str)
    args = parser.parse_args()

    wandb.login(key=args.wandb_key)
    logger = WandbLogger(project="AutoEncoder")
    model = AutoEncoder()

    dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    train_loader = DataLoader(dataset)

    trainer = Trainer(limit_train_batches=100, max_epochs=1, accelerator='gpu', logger=logger)
    trainer.fit(model, train_loader)
    wandb.finish()