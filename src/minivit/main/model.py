from typing import cast

import torch
from torch import nn

from minivit.models.config import (
    ArchitectureConfig,
    AttentionConfig,
    PatchEmbedderConfig,
)


class TanhGate(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.tanh(x) + self.beta


class PatchEmbedder(nn.Module):
    def __init__(self, *, config: PatchEmbedderConfig) -> None:
        super().__init__()
        assert config.image_size % config.patch_size == 0
        self.patch_transform = nn.Conv2d(
            config.number_of_input_channels,
            config.embedding_dimension,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.class_token = nn.Parameter(torch.randn(1, 1, config.embedding_dimension))
        number_of_patches = (config.image_size // config.patch_size) ** 2
        number_of_tokens = number_of_patches + 1  # class token
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, number_of_tokens, config.embedding_dimension),
        )
        self.normalization = TanhGate(config.embedding_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # channel, height, width
        embedded_patches = self.patch_transform(x)
        # channel, embedding_dimension, height / patch_size, width / patch_size
        flattened_patches = embedded_patches.flatten(2).transpose(1, 2)
        # number_of_patches, embedding_dimension
        batch_size = len(flattened_patches)
        class_token = self.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat((class_token, flattened_patches), dim=1)
        tokens = tokens + self.positional_embeddings
        return cast(torch.Tensor, self.normalization(tokens))


class SelfAttentionBlock(nn.Module):
    def __init__(self, *, config: AttentionConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.embedding_dimension,
            config.number_of_heads,
            batch_first=True,
        )
        self.normalization = TanhGate(config.embedding_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized_x = self.normalization(x)
        attention_inputs = normalized_x, normalized_x, normalized_x  # self-attention
        attention_outputs, _ = self.attention(*attention_inputs)
        return cast(torch.Tensor, self.normalization(attention_outputs))


class MLPBlock(nn.Module):
    def __init__(self, *, embedding_dimension: int, mlp_dimension: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dimension, mlp_dimension),
            nn.GELU(),
            nn.Linear(mlp_dimension, embedding_dimension),
        )
        self.normalization = TanhGate(embedding_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.layers(x)
        return cast(torch.Tensor, self.normalization(output))


class TransformerBlock(nn.Module):
    def __init__(self, *, config: AttentionConfig) -> None:
        super().__init__()
        self.attention = SelfAttentionBlock(config=config)
        self.mlp = MLPBlock(
            embedding_dimension=config.embedding_dimension,
            mlp_dimension=config.mlp_dimension,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        return cast(torch.Tensor, x + self.mlp(x))


class TransformerEncoder(nn.Module):
    def __init__(self, *, config: ArchitectureConfig) -> None:
        super().__init__()
        self.patch_embedder = PatchEmbedder(config=config.patch_embedder)
        blocks = [
            TransformerBlock(config=config.attention) for _ in range(config.depth)
        ]
        self.blocks = nn.Sequential(*blocks)
        self.normalization = TanhGate(config.embedding_dimension)
        self.classification_head = nn.Linear(
            config.embedding_dimension,
            config.number_of_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embedder(x)
        return cast(torch.Tensor, tokens + self.blocks(tokens))


class TransformerClassifier(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(config=config)
        self.classification_head = nn.Linear(
            config.embedding_dimension,
            config.number_of_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encodings = self.encoder(x)
        encoding = encodings[:, 0]  # Take CLS token
        return cast(torch.Tensor, self.classification_head(encoding))
