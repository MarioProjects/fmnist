"""
Getters for loss functions.

Author: Mario Parreño Lara (maparla@proton.me)

Date: 07/10/2023
"""
from argparse import Namespace
import torch


def get_criterion(config: Namespace) -> torch.nn.Module:
    """Returns the criterion based on the config

    Args:
        config (Namespace): The config object
        
    Returns:
        criterion (torch.nn.Module): The criterion
    """
    if config.loss == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Loss not supported")
