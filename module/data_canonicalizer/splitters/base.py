from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class BaseSplitter:
    name = "base"

    def split(self, data: np.ndarray, args, meta: dict[str, Any]) -> SplitResult:
        raise NotImplementedError
