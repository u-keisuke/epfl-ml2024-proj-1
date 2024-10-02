import random

import numpy as np
import torch


def set_random_seed(seed: int | None = 0):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)