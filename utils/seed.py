from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """固定随机种子，便于复现实验。"""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
