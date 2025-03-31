from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar

from minivit.context import context

from .data import MNISTDataModule
from .model import TransformerClassifier
from .trainer import TrainerModule


def main() -> None:
    """
    Train mini Vision Transformer for MNIST character recognition.
    """
    model = TransformerClassifier(config=context.config.architecture)
    trainer_module = TrainerModule(model, learning_rate=context.config.learning_rate)
    data_module = MNISTDataModule()
    trainer = Trainer(
        max_epochs=context.config.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[RichProgressBar()],
    )
    trainer.fit(trainer_module, datamodule=data_module)
