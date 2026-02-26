"""Device utilities for cross-platform support."""

import os
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and os.environ.get("FLOWREG_ENABLE_MPS", "0") == "1":
        return torch.device("mps")
    return torch.device("cpu")
