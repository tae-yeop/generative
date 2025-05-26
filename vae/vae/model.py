import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/pytorch/examples/tree/master/vae
# https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py
class VAE(nn.Module):
    def __init__(
        self,
        in_dim=784, 
        hidden_dim=400, 
        latent_dim=20
    ):
        super().__init__()

        # Encoder
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, in_dim)
        

    def encode(self, x):
        """
        part of q_phi(z|x) : mu(x), log sigma^2(x)
        """
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)


     def reparameterize(self, mu, logvar):
        """
        part of q_phi(z|x) : z = mu + sigma * epsilon
        """
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def decode(self, z):
        """
        p_theta(x|z)
        """
        h = torch.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        """        
        q_phi(z|x) : mu(x), log sigma^2(x)
        p_theta(x|z)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    @torch.no_grad()
    def sample(self, n_samples=64):
        """
        p_theta(x|z)
        """
        z = torch.randn(n_samples, self.latent_dim)
        samples = self.decode(z)
        return samples
    

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),   # -> (B,64,32,32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (B,128,16,16)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# -> (B,256,8,8)
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),# -> (B,256,4,4)
            nn.ReLU(),
        )

        # 256*4*4 = 4096 -> latent
        self.fc_mu = nn.Linear(4096, latent_dim)
        self.fc_logvar = nn.Linear(4096, latent_dim)


        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 4096)
        self.dec_deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # -> (B,256,8,8)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> (B,128,16,16)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # -> (B,64,32,32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),     # -> (B,3,64,64)
            nn.Sigmoid()  # 픽셀값 (0,1)
        )

    def encode(self, x):
        """
        part of q_phi(z|x) : mu(x), log sigma^2(x)
        """
        h = self.enc_conv(x)           # (B,256,4,4)
        h = h.view(h.size(0), -1)      # (B,4096)
        mu = self.fc_mu(h)             # (B, latent_dim)
        logvar = self.fc_logvar(h)     # (B, latent_dim)
        return mu, logvar


    def reparameterize(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def decode(self, z):
        """
        p_theta(x|z)
        """
        h = self.dec_fc(z)  # (B,4096)
        h = h.view(-1, 256, 4, 4)  # (B,256,4,4)
        x_recon = self.dec_deconv(h)  # (B,3,64,64
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    @torch.no_grad()
    def sample(self, n_samples=64):
        """
        p_theta(x|z)
        """
        z = torch.randn(n_samples, self.latent_dim)
        samples = self.decode(z)
        return samples



# Stable Diffusion VAE

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels if out_channels else in_channels
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
    
    def forward(self, x):
        return self.deconv(x)

class ResBlock(nn.Module):
    """
    간단한 Residual Block 예시:
      - Conv -> GroupNorm -> activation -> Conv -> GroupNorm -> skip connect
    실제 Stable Diffusion 코드는 더 복잡(동적 채널 변환, Swish/GELU 활성화, etc.)
    """


class DownB