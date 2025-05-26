import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import numpy as np
import yaml
import argparse
from omegaconf import OmegaConf

from packaging import version
import lightning as L
from lightning import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.strategies import DDPStrategy
from lightning import seed_everything


import argparse
import importlib
import os

def is_lightning_v2():
    return version.parse(L.__version__) >= version.parse("2.0.0")

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    # model.traget이 되기 떄문에 cldm.cldm.ControlLDM가 됨
    # Model class : cldm.cldm.ControlLDM 여기에 params을 kwargs형태로 __init__에 넣는 꼴
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_state_dict(ckpt_path, device):
    """
    safetensors 파일까지 처리하는 코드

    load_file(path) :  가장 간단하고 PyTorch 스타일로 state_dict 바로 얻고 싶을 때
    safe_open(path)	: 큰 모델에서 일부만 로드하거나, 메모리 최적화할 때
    load(data): 메모리에서 바로 로딩할 때 (예: 서버에서 받은 바이트 데이터)
    """
    def get_state_dict(d):
        return d.get('state_dict', d)

    _, extension = os.path.splitext(ckpt_path)

    if extension.lower() == ".safetensors":
        try:
            import safetensors.torch
        except ImportError:
            raise ImportError("Please install safetensors to load .safetensors checkpoints.")
        state_dict = safetensors.torch.load_file(ckpt_path, device=device)
    else:
        loaded = torch.load(ckpt_path, map_location=device)
        state_dict = get_state_dict(loaded)

    return state_dict

def wrap_kwargs(f):
    sig = inspect.signature(f)
    # Check if f already has kwargs
    has_kwargs = any([
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    ])
    if has_kwargs:
        @wraps(f)
        def f_kwargs(*args, **kwargs):
            y = f(*args, **kwargs)
            if isinstance(y, tuple) and isinstance(y[-1], dict):
                return y
            else:
                return y, {}
    else:
        param_kwargs = inspect.Parameter("kwargs", kind=inspect.Parameter.VAR_KEYWORD)
        sig_kwargs = inspect.Signature(parameters=list(sig.parameters.values())+[param_kwargs])
        @wraps(f)
        def f_kwargs(*args, **kwargs):
            bound = sig_kwargs.bind(*args, **kwargs)
            if "kwargs" in bound.arguments:
                kwargs = bound.arguments.pop("kwargs")
            else:
                kwargs = {}
            y = f(**bound.arguments)
            if isinstance(y, tuple) and isinstance(y[-1], dict):
                return *y[:-1], {**y[-1], **kwargs}
            else:
                return y, kwargs
    return f_kwargs

def discard_kwargs(f):
    if f is None: return None
    f_kwargs = wrap_kwargs(f)
    @wraps(f)
    def f_(*args, **kwargs):
        return f_kwargs(*args, **kwargs)[0]
    return f_


class MNISTDataset(LightningDataModule):
    def __init__(
            self,
            data_dir="./data",
            batch_size: int = 64,
            num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
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
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, 
            num_workser=self.num_workers, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)


class GAN(LightningModule):
    """
    - lightning에선 장치 설정을 nn.Module을 상속하는한 알아서 해준다
    - LightningModule은 nn.Module을 상속했음
    - LightningModule에서 쓰는 self.logger는 사실 Trainer 밖에서 만들고 전달받은걸 사용
    """
    def __init__(self, args):
        """
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
        """
        super().__init__()
        self.config = args
        self.build_model()
        self.setup_val_metrics()
        self.automatic_optimization = False
        self.validation_z = torch.randn(8, self.hparams.latent_dim, device=self.device)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def build_model(self):
        self.generator = instantiate_from_config(self.config.generator)
        self.discriminator = instantiate_from_config(self.config.discriminator)
        self.objective_model =  instantiate_from_config(self.config.objective)

        if self.config.resume_path:
            self.generator.load_state_dict(load_state_dict(self.training_config.resume_path,
                                                       location='cpu'))
            
            self.discriminator.load_state_dict(load_state_dict(self.training_config.resume_path,
                                                       location='cpu'))
    def configure_optimizers(self):
        optim_type = self.config.optimization.optimizer
        generator_lr = self.config.optimization.generator_lr
        discriminator_lr = self.config.optimization.discriminator_lr

        params_g = list(self.generator.parameters())
        params_d = list(self.discriminator.parameters())

        optimizer_g = discard_kwargs(torch.optim.Adam)(net=self.generator, params=params_g, lr=generator_lr)
        optimizer_d = discard_kwargs(torch.optim.Adam)(net=self.discriminator, params=params_d, lr=discriminator_lr)
        return [optimizer_g, optimizer_d], []
        
    def setup_val_metrics(self):
        if self.config.validation is None:
            return None
        metrics = {}
        for name, target_dict in self.config.validation.metrics.items():
            metrics[name] = instantiate_from_config(target_dict)

    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, pred, target_is_real=True):
        # target tensor
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        loss = F.binary_cross_entropy(pred, target)
        return loss

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()

        real_imgs, _ = batch
        z = torch.randn(batch.size(0), self.hparams.latent_dim, device=self.device)
        fake_imgs = self(z)
        grid = torchvision.utils.make_grid(fake_imgs)
        self.logger.log_image("name", [grid,])

        self.discriminator_step(fake_imgs.detach(), real_imgs, optimizer_d)
        self.generator_step(fake_imgs, real_imgs, optimizer_g)


    def discriminator_step(self, fake_imgs, real_imgs, optimizer_d):

        # Discriminator Forward-Backward
        input_disc = torch.concat([fake_imgs, real_imgs])
        logits =  self.discriminator(input_disc)
        real_logit, fake_logit = logits.chunk(2, dim=0)
        d_loss = self.objective_model.loss_d(real_logit, fake_logit)
        self.log('d_loss', d_loss, prog_bar=True)

        if is_lightning_v2():
            with self.optimizers().optimizer_context(optimizer_d):
                self.manual_backward(d_loss)
                optimizer_d.step()
                optimizer_d.zero_grad()
        else:
            # fix global_step in GAN training
            # https://github.com/Lightning-AI/pytorch-lightning/issues/17958
            self.toggle_optimizer(optimizer_d)
            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()


        # 정규화 loss (D regularization)
        if self.global_step % self.config.optimization.d_reg_freq == 0:
            real_imgs.requires_grad = True
            real_logit = self.discriminator(real_imgs)
            reg_d_loss = self.objective_model.regularize_d(real_logit, real_imgs    ) * self.config.optimization.d_reg_freq
            self.log('reg_d_loss', reg_d_loss, prog_bar=True)

            if is_lightning_v2():
                with self.optimizers().optimizer_context(optimizer_d):
                    self.manual_backward(reg_d_loss)
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            else:
                self.toggle_optimizer(optimizer_d)
                self.manual_backward(reg_d_loss)
                optimizer_d.step()
                optimizer_d.zero_grad()
                self.untoggle_optimizer(optimizer_d)

        return d_loss
    
    def generator_step(self, fake_imgs, real_imgs, optimizer_g):
        fake_logit = self.discriminator(fake_imgs)
        g_loss, g_loss_dict = self.objective_model.loss_g(fake_logit, fake_imgs, real_imgs)
        self.log_dict(g_loss_dict, prog_bar=True)

        if is_lightning_v2():
            with self.optimizers().optimizer_context(optimizer_g):
                self.manual_backward(g_loss)
                optimizer_g.step()
                optimizer_g.zero_grad()
        else:
            # fix global_step in GAN training
            # https://github.com/Lightning-AI/pytorch-lightning/issues/17958
            self.toggle_optimizer(optimizer_g)
            self.manual_backward(g_loss)
            optimizer_g.step()
            optimizer_g.zero_grad()
            self.untoggle_optimizer(optimizer_g)

        return g_loss
    
    def validation_step(self, batch, batch_idx):
        """
        최신 라이트닝 : 자동으로 eval mode + no_grad 해준다
        """
        real, _ = batch  # real images
        batch_size = real.size(0)

        # fixed noise for visualization
        z = self.validation_z# shape: [8, latent_dim] or more
        fake = self(z)

        # 이미지 그리드 생성 (real / fake 비교)
        real_grid = torchvision.utils.make_grid(real[:8], nrow=4, normalize=True, value_range=(-1, 1))
        fake_grid = torchvision.utils.make_grid(fake[:8], nrow=4, normalize=True, value_range=(-1, 1))

        self.logger.log_image("samples", images=[real_grid, fake_grid], caption=["real", "fake"])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="number of available GPUs")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="path to checkpoint_dir")
    parser.add_argument("--max_steps", type=int, default=25000, help="training step")
    parser.add_argument("--val_freq", type=int, default=0, help="check validation every n train batches")
    parser.add_argument("--config_path", type=str, default=None, help="training model configuration path")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--num_nodes", type=int, default=2, help="num nodes")    
    parser.add_argument("--deterministic", action='store_true', help="reproducibility")
    parser.add_argument("--resume_path", type=str, default=None, help="resume checkpoint path")
    parser.add_argument('--config', type=str, default='configs/cfg_example.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    assert args.config is not None
    args = OmegaConf.load(args.config)
    OmegaConf.set_struct(args, False)

    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
    pl_trainer_cfg = dict(
        accelerator="gpu", 
        devices=args.gpus, 
        precision=32, 
        num_nodes=args.num_nodes, 
        strategy=ddp,
        logger=wandb_logger if wandb_logger is not None else None,
        max_steps=args.max_steps, 
        val_check_interval=args.val_freq
    )

    if args.deterministic:
        seed_everything(1, workers=True)
        trainer_cfg.update({'deterministic': True})

    # -----------------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------------
    data_module = MNISTDataModule(batch_size=args.batch_size)

    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    gan = GAN(*data_module.dims)
    if args.resume_path:
        coach_model.load_state_dict(load_state_dict(args.resume_path, location='cpu'))

    # -----------------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------------
    trainer = Trainer(**trainer_cfg)
    trainer.fit(model, data_module)