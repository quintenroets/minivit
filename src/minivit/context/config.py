from dataclasses import dataclass, field


@dataclass
class AttentionConfig:
    number_of_heads: int = 4
    embedding_dimension: int = 64
    mlp_dimension: int = 128


@dataclass
class PatchEmbedderConfig:
    image_size: int = 28
    patch_size: int = 7
    number_of_input_channels: int = 1
    embedding_dimension: int = 64


@dataclass
class ArchitectureConfig:
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    patch_embedder: PatchEmbedderConfig = field(default_factory=PatchEmbedderConfig)
    depth: int = 5
    number_of_classes = 10

    @property
    def embedding_dimension(self) -> int:
        return self.attention.embedding_dimension

    def __post_init__(self) -> None:
        assert self.embedding_dimension == self.patch_embedder.embedding_dimension


@dataclass
class Config:
    max_epochs: int = 5
    learning_rate: float = 1e-3
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
