import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

def g_logistic_loss(fake_pred):
    return F.softplus(-fake_pred).mean()

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

def g_hinge(d_logit_fake):
    return -torch.mean(d_logit_fake)

def d_hinge(d_logit_real, d_logit_fake):
    return torch.mean(F.relu(1. - d_logit_real)) + torch.mean(F.relu(1. + d_logit_fake))

def d_r1_loss(real_logit, real_img, r1_gamma):
    """
    (gradient penalty) for the discriminator when using real images.
    real image에 대해 discriminator가 너무 가파른(값이 급격히 변하는) 
    결괏값을 내지 않도록 만들어 주는 정규화(regulazation) 역할을 합니다.

    Args:
        real_logit (torch.Tensor): Discriminator가 real image에 대해 출력한 logit 값.
            shape = [batch_size, 1] 또는 [batch_size].
        real_img (torch.Tensor): 실제 이미지 텐서.
            예: shape = [batch_size, channels, height, width].
        r1_gamma (float): R1 penalty 계수(강도). 
        
    Returns:
        torch.Tensor: R1 페널티 값. (스칼라 값)
    """
    # real_logit.sum()을 목적함수로 설정하고, real_img에 대한 gradient를 구함.
    # grad_real의 shape = [batch_size, channels, height, width].
    grad_real, = grad(outputs=real_logit.sum(),       # 스칼라 값 (모든 배치 logit 합)
                      inputs=real_img,                # 실제 이미지 텐서
                      create_graph=True)              # 2차 미분 등을 할 수 있도록 그래프 생성
    

    # grad_real.pow(2)로 각 픽셀/채널 차원을 제곱.
    # .reshape(grad_real.shape[0], -1) : (batch_size, channels*height*width)로 변형.
    # .sum(1): 각 배치마다 모든 픽셀/채널 요소를 합산.
    # .mean(): 배치 차원을 다시 평균냄.
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    # 최종 R1 손실: 0.5 * r1_gamma * grad_penalty
    return 0.5 * r1_gamma * grad_penalty