from typing import cast

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import Accuracy

from minivit.context import context


class TrainerModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.model = model
        self.loss_function = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        stages = "train", "validation"
        self.accuracies = {
            stage: Accuracy(task="multiclass", num_classes=10).to(context.device)
            for stage in stages
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast("torch.Tensor", self.model(x))

    def training_step(
        self,
        batch: tuple[torch.Tensor, ...],
        batch_index: int,  # noqa: ARG002
    ) -> torch.Tensor:
        return self.calculate_loss(batch, "train")

    def validation_step(
        self,
        batch: tuple[torch.Tensor, ...],
        batch_index: int,  # noqa: ARG002
    ) -> None:
        self.calculate_loss(batch, "validation")

    def calculate_loss(
        self,
        batch: tuple[torch.Tensor, ...],
        stage: str,
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)
        accuracy = self.accuracies[stage](logits, y)
        self.log(f"{stage}_loss", loss)
        self.log(f"{stage}_accuracy", accuracy, prog_bar=True)
        return cast("torch.Tensor", loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
