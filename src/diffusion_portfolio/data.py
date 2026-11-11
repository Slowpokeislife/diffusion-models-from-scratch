"""Dataset loading and normalization to [-1,1]."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loader(dataset_name, data_root='data', batch_size=64, size=32, workers=0, max_samples=None):
    tx = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * (3 if dataset_name == 'flowers' else 1),
                             (0.5,) * (3 if dataset_name == 'flowers' else 1))
    ])
    if dataset_name == 'mnist':
        ds = datasets.MNIST(data_root, train=True, transform=tx, download=True)
    elif dataset_name == 'fashion':
        ds = datasets.FashionMNIST(data_root, train=True, transform=tx, download=True)
    elif dataset_name == 'flowers':
        folder = Path(data_root) / 'cropped_flowers'
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing {folder}. Extract your original flower data as described in README.')
        ds = datasets.ImageFolder(str(folder), transform=tx)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    if max_samples is not None:
        ds = torch.utils.data.Subset(ds, range(min(max_samples, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=False), ds
