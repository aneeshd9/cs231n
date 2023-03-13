from typing import Dict, Tuple
import numpy as np


def sgd(w: np.ndarray, dw: np.ndarray,
        config: Dict[str, float] | None = None) -> Tuple[np.ndarray, Dict[str, float]]:
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config
