import torch
from torch import nn as nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def cosine_beta_schedule(timesteps, s=0.008):
    """
    IDDPM
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def constant_beta_schedule(timesteps):
    scale = 1000 / timesteps
    constant = scale * 0.01
    return torch.tensor([constant] * timesteps, dtype=torch.float64)

def draw():
    timesteps = 100

    # 각 스케줄 함수로부터 베타 값 생성
    linear_betas = linear_beta_schedule(timesteps)
    quadratic_betas = quadratic_beta_schedule(timesteps)
    #cosine_betas = cosine_beta_schedule(timesteps)
    sigmoid_betas = sigmoid_beta_schedule(timesteps)

    # 그래프 그리기
    plt.figure(figsize=(10, 5))
    plt.plot(linear_betas, label='Linear Beta Schedule')
    plt.plot(quadratic_betas, label='Quadratic Beta Schedule')
    # plt.plot(cosine_betas, label='Cosine Beta Schedule')
    plt.plot(sigmoid_betas, label='Sigmoid Beta Schedule')
    plt.title('Comparison of Linear and Quadratic Beta Schedules')
    plt.xlabel('Timestep')
    plt.ylabel('Beta value')
    plt.legend()
    plt.grid(True)

    # 그래프를 PNG 파일로 저장
    plt.savefig('beta_schedules.png')

timesteps = 100
betas = cosine_beta_schedule(timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
print(alphas_cumprod_prev)
timesteps, = betas.shape
print(timesteps)
# self.num_timesteps = int(timesteps)
# self.extand_noise_spans = extand_noise_spans