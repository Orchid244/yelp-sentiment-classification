import json
import os
import random

import numpy as np
import torch

from config import (
    BASELINE_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    RESULT_DIR,
    SPLIT_DIR,
    TRANSFORMER_MODEL_DIR,
    TRANSFORMER_RESULT_DIR,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def ensure_dirs() -> None:
    for path in [
        OUTPUT_DIR,
        MODEL_DIR,
        RESULT_DIR,
        SPLIT_DIR,
        BASELINE_DIR,
        TRANSFORMER_MODEL_DIR,
        TRANSFORMER_RESULT_DIR,
    ]:
        os.makedirs(path, exist_ok=True)



def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)



def save_text(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)