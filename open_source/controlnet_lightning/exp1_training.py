from share import *
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from exp1_dataset import DFDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning import seed_everything

seed_everything(1)

def get_model_size(model):
    params = list(model.parameters())
    buffers = list(model.buffers())

    size_bytes = sum(np.prod(param.size()) * param.element_size() for param in params)
    size_bytes += sum(np.prod(buffer.size()) * buffer.element_size() for buffer in buffers)

    size_mb = size_bytes / (1024 * 1024)
    print('model size: {:.3f}MB'.format(size_mb))
    return size_mb

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15_exp1.yaml').cpu()
print(type(model.model.diffusion_model), get_model_size(model.model.diffusion_model))
# model.load_state_dict(load_state_dict(resume_path, location='cpu'))
# model.learning_rate = learning_rate
# model.sd_locked = sd_locked
# model.only_mid_control = only_mid_control


# # Misc
# dataset = DFDataset()
# dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
# logger = ImageLogger(batch_frequency=logger_freq)
# # trainer = pl.Trainer(accelerator="gpu", strategy='ddp', devices=8, precision=32, callbacks=[logger])
# # 원본
# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
# # trainer = pl.Trainer(accelerator="gpu", strategy='ddp', devices=8, precision=32, callbacks=[logger])

# # Train!
# trainer.fit(model, dataloader)
