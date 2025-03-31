from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()
        self.train: datasets.MNIST = None
        self.validation: datasets.MNIST = None

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        self.train = datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=self.transform,
        )
        self.validation = datasets.MNIST(
            "./data",
            train=False,
            download=True,
            transform=self.transform,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(self.validation, batch_size=self.batch_size)
