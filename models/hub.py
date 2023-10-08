"""
Functions to get models from the timm library

Author: Mario Parreño Lara (maparla@proton.me)

Date: 07/10/2023
"""
from argparse import Namespace
import timm
import torch
from etl.dataset import IN_CHANNELS, NUM_CLASSES


AVAILABLE_TIMM_MODELS = timm.list_models()


def get_model(config: Namespace) -> torch.nn.Module:
    """Returns the model based on the config

    Args:
        config (Namespace): The config object

    Returns:
        model (torch.nn.Module): The model
    """
    if config.weights == "random":
        pretrained = False
    elif config.weights == "imagenet":
        pretrained = True
    else:
        raise ValueError(f"Weights '{config.weights}' not supported")

    if config.model in AVAILABLE_TIMM_MODELS:
        return timm.create_model(
            config.model,
            pretrained=pretrained,
            in_chans=IN_CHANNELS,
            num_classes=NUM_CLASSES
        )
    else:
        raise ValueError("Model not supported")
