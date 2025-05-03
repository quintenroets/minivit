from functools import cached_property

import torch
from package_utils.context import Context as Context_

from .config import Config
from .options import Options
from .secrets_ import Secrets


class Context(Context_[Options, Config, Secrets]):
    @cached_property
    def device(self) -> torch.device:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device_name)


context = Context(Options, Config, Secrets)
