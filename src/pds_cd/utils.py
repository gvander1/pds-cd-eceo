import random
from contextlib import contextmanager

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@contextmanager
def fixed_rng(seed):
    """Run a block with a fixed torch seed, then restore the previous RNG state.

    Used around evaluation: the random Hi-UCD val crops are then identical at every
    epoch and in every run, and training randomness (dropout, crops) is left untouched.
    """
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield
