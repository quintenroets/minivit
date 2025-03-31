from functools import cached_property

import torch
from package_utils.context import Context as Context_

from minivit.models import Config, Options, Secrets


class Context(Context_[Options, Config, Secrets]):
    @cached_property
    def device(self) -> torch.device:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device_name)


context = Context(Options, Config, Secrets)
