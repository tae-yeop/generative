import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from tqdm import tqdm
import matplotlib.pyplot as plt


class VAELoss(nn.Module):
    def __init__(self, kl_coef=0.01):
        super().__init__()
        self.kl_coef = kl_coef
        self.bce = 0
        self.kl = 0

    def forward(self, x, recon_x, mu, logvar):
        self.bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # see Appendix B from VAE paper:
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return self.bce + self.kl_coef*self.kl
        


class UTKFaceWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]  # {'image': PIL Image, 'age': int, 'gender': int, 'ethnicity': int, ...}
        img = sample["image"]
        if self.transform:
            img = self.transform(img)  # (3, H, W)
        return img  # VAE 입력용, 레이블은 사용 안 함


if __name__ == "__main__":
    raw_dset = load_dataset("utk_face")
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(), 
    ])

    train_dataset = UTKFaceWrapper(raw_dset["train"], transform=transform)
    test_dataset  = UTKFaceWrapper(raw_dset["test"],  transform=transform)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)



    model = SimpleConvVAE(latent_dim=128).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    vae_loss = VAELoss()

    epochs = 5

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{epochs}")):
            data = data.cuda()

            x_recon, mu, logvar = model(data)
            loss = vae_loss(data, x_recon, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss = {avg_loss:.4f}")
            
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(test_loader, desc=f"[Test] Epoch {epoch+1}/{epochs}")):
                data = data.cuda()
                x_recon, mu, logvar = model(data)
                loss = vae_loss(data, x_recon, mu, logvar)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Test Loss = {avg_test_loss:.4f}")


    # sampling
    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples=16).cpu()
        grid_tensor = make_grid(samples, nrow=4, padding=2)
        grid_image = to_pil_image(grid_tensor.cpu())  # CPU로 옮겨 PIL 변환
        