from dataclasses import dataclass

from minivit.models.path import Path


@dataclass
class Options:
    config_path: Path = Path.config
