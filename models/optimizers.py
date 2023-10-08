"""
Getters for optimizers.

Author: Mario Parreño Lara (maparla@proton.me)

Date: 07/10/2023
"""
from pytorch_lightning import LightningModule
import torch


def get_optimizer(pl_module: LightningModule) -> torch.optim.Optimizer:
    """Returns the optimizer based on the pytorch lightning module config

    Args:
        pl_module (LightningModule): The pytorch lightning module

    Returns:
        optimizer (torch.optim.Optimizer): The optimizer
    """
    if pl_module.optimizer == "SGD":
        return torch.optim.SGD(
            pl_module.parameters(),
            lr=pl_module.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
    else:
        raise ValueError("Optimizer not supported")