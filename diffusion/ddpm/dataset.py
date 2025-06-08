from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision import transforms

import torch
import torchvision
from datasets import load_dataset

import os
from pathlib import Path
from PIL import Image
from io import BytesIO

from tqdm import tqdm

import random

def pil_from_row(row):
    if "image" in row:
        # Image feature이면 이미 PIL.Image 또는 numpy array가 됨
        return row["image"] if isinstance(row["image"], Image.Image) else Image.fromarray(row["image"])

    elif "bytes" in row:
        return Image.open(BytesIO(row["bytes"]))

    else:
        raise KeyError("row에 'image' 또는 'bytes' 열이 없습니다.")

class HuggingDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]['image']
        img = self.transform(item)

        return img 

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = list(Path(root).rglob('*.jpg')) + list(Path(root).rglob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = Image.open(self.dataset[idx])
        if self.transform:
            img = self.transform(img)

        return img


class ImageOnlyDataset(Dataset):
    """
    MNIST가 label도 같이 내놓기 때문에 필요한 wrapper
    파이썬 루프가 있는 collate_fn을 따로 구현하는것 보다 C++ 레벨에서 돌아가도록 
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img


def get_loader(name, image_size, batch_size, save_image=False):
    if name == 'mnist':
        transform = Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - 0.5) *2) # 초기 범위 [0,1]
        ])

        os.makedirs('./data/mnist', exist_ok=True)
        raw_trainset = torchvision.datasets.MNIST(
            root='data/mnist', download=True, transform=transform
        )

        # def collate_fn(batch):
        #     img_list = []
        #     for x, _ in batch:
        #         img_list.append(x)

        #     new_batch = torch.stack(img_list)
        #     return new_batch

        # from torch.utils.data._utils.collate import default_collate
        # better_collate_fn = lambda batch: default_collate([img for img, _ in batch])

        # train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=better_collate_fn)

        trainset = ImageOnlyDataset(raw_trainset)

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    elif name == 'pokemon':
        split = "train[:10]"
        streaming = False
        raw_dataset = load_dataset(
            "huggan/pokemon",
            split=split,
            streaming=streaming,
            # features=Features({"image": Image()})  # bytes → 이미지 feature 자동 변환(가능할 때)
        )

        if save_image:
            os.makedirs('./data/pokemon', exist_ok=True)
            for idx, row in tqdm(enumerate(raw_dataset), desc="saving"):
                img = pil_from_row(row)
                img.save(f"./data/pokemon/{idx:05d}.png")

            transform = Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x - 0.5) *2) # 초기 범위 [0,1]
            ])

            trainset = ImageDataset('./data/pokemon', transform)

            train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        else:
            transform = Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x - 0.5) *2) # 초기 범위 [0,1]
            ])

            trainset = HuggingDataset(raw_dataset, transform)

            train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)


    return train_loader, trainset

if __name__ == '__main__':
    import numpy as np
    train_loader, train_set = get_loader('pokemon', 128, 16, save_image=True)

    samples = random.choices(train_set, k=16)

    # print(samples[0])
    # print(type(samples[0]), samples[0].shape)

    for x in train_loader:
        print(x.shape)
        break


    print(x[0].shape)
    from torchvision.utils import save_image
    save_image((x[0] + 1) / 2, "pic.png")