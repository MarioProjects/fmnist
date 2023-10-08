"""
Normalization functions for images

Author: Mario Parreño Lara (maparla@proton.me)

Date: 07/10/2023
"""
import numpy as np


def apply_normalization(
    image: np.ndarray,
    normalization_type: str,
    mean: float|None = None,
    std: float|None = None,
    image_min: float|None = None,
    image_max: float|None = None,
) -> np.ndarray:
    """Applies the normalization to the image
    
    Args:
        image (np.ndarray): The image to normalize
        normalization_type (str): The normalization type
        mean (float): The mean to use for standardization
        std (float): The std to use for standardization
        image_min (float): The min value to use for reescale
        image_max (float): The max value to use for reescale

    Returns:
        image (np.ndarray): The normalized image
    """
    if normalization_type == "none" or normalization_type is None:
        return image
    elif normalization_type == "255":
        return image / 255.0
    elif normalization_type == "reescale":
        if image_min is None:
            image_min = image.min()
        if image_max is None:
            image_max = image.max()
        image = (image - image_min) / (image_max - image_min)
        return image
    elif normalization_type == "negative1_positive1":
        # https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
        image = 2 * ((image - image.min()) / (image.max() - image.min())) - 1
        return image
    elif normalization_type == "standardize":
        if mean is None:
            mean = np.mean(image)
        if std is None:
            std = np.std(image)
        image = image - mean
        image = image / std
        return image
    assert False, "Unknown normalization: '{}'".format(normalization_type)
