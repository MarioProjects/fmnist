"""
Augmentations getter used for the training and validation

Author: Mario Parreño Lara (maparla@proton.me)

Date: 07/10/2023
"""
from argparse import Namespace
import albumentations as A


def get_transforms(run_config: Namespace) -> tuple[A.Compose, A.Compose]:
    """Returns the train and validation transforms based on the run config

    Args:
        run_config (Namespace): The run config object

    Returns:
        train_transform, val_transform (A.Compose): The train and validation transforms
    """

    train_transform = A.Compose([
        A.HorizontalFlip(p=run_config.p_hflip),
        A.VerticalFlip(p=run_config.p_vflip),
        A.Rotate(
            p=run_config.p_rotate, limit=(-75, 75),
            interpolation=0, border_mode=0, value=(0, 0, 0),
            mask_value=None, crop_border=False
        ),
        A.Posterize(p=run_config.p_posterize, num_bits=[2, 5]),
        A.RandomBrightnessContrast(  # Brightness controller
            p=run_config.p_brightness,
            brightness_limit=(0), contrast_limit=(-0.5, 0.5)
        ),
        A.RandomBrightnessContrast(  # Contrast controller
            p=run_config.p_contrast,
            brightness_limit=(-0.5, 0.5),
            contrast_limit=(0),
            brightness_by_max=True
        ),
        A.AdvancedBlur(
            p=run_config.p_blur,
            blur_limit=(3, 7),
            sigmaX_limit=(0.2, 1.0),
            sigmaY_limit=(0.2, 1.0),
            rotate_limit=(-90, 90),
            beta_limit=(0.5, 8.0),
            noise_limit=(0.9, 1.1)
        ),
        A.CoarseDropout(
            p=run_config.p_dropout,
            min_holes=3,
            max_holes=6,
            min_height=4,
            max_height=6,
            min_width=4,
            max_width=6,
            fill_value=0
        ),
        A.ShiftScaleRotate(
            p=run_config.p_shift,
            shift_limit_x=(-0.25, 0.25),
            shift_limit_y=(-0.25, 0.25),
            scale_limit=0,
            rotate_limit=0,
            interpolation=0, border_mode=0, value=0, mask_value=None
        ),
        A.RandomResizedCrop(
            p=run_config.p_random_crop,
            height=28,
            width=28,
            scale=(0.75, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation=0
        )
    ])

    val_transform = A.Compose([])

    return train_transform, val_transform
