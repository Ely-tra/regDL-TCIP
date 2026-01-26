from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class CanonicalizedData:
    data: np.ndarray
    meta: dict[str, Any]


class BaseCanonicalizer:
    name = "base"

    def supported_splitters(self) -> Sequence[str]:
        return ()

    def default_splitter(self, args) -> str:
        supported = self.supported_splitters()
        if not supported:
            raise ValueError(f"{self.name} has no supported splitters")
        return supported[0]

    def load(self, args) -> CanonicalizedData:
        raise NotImplementedError
