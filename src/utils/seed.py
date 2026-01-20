"""
Random seed utilities for reproducibility.

Ensures deterministic runs across numpy, torch, and python random.
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Create a torch Generator with optional seed.

    Args:
        seed: Optional seed for the generator.

    Returns:
        Seeded torch Generator.
    """
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator
