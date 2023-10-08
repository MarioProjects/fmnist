"""
Getters for schedulers.

Author: Mario Parreño Lara (maparla@proton.me)

Date: 07/10/2023
"""
import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import MultiStepLR


def get_scheduler(
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler:
    """Returns the scheduler based on the pytorch lightning module config

    Args:
        pl_module (LightningModule): The pytorch lightning module
        optimizer (torch.optim.Optimizer): The optimizer to apply the scheduler

    Returns:
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler
    """
    if pl_module.scheduler == "MultiStepLR":
        return MultiStepLR(
            optimizer,
            milestones=pl_module.milestones,
            gamma=pl_module.step_gamma
        )
    else:
        raise ValueError("Scheduler not supported")