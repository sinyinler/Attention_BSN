from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from utils.image_io import load_normalized_image


@dataclass
class SingleImageData:
    """单张 BFI 图的数据容器。"""

    path: Path
    image: np.ndarray
    norm_meta: Dict[str, Any]


def load_single_image_data(path: str | Path, data_config: Dict[str, Any]) -> SingleImageData:
    image, meta = load_normalized_image(
        path,
        mode=data_config.get("normalize", "percentile"),
        percentile_low=float(data_config.get("percentile_low", 1.0)),
        percentile_high=float(data_config.get("percentile_high", 99.0)),
    )
    return SingleImageData(path=Path(path), image=image, norm_meta=meta)
