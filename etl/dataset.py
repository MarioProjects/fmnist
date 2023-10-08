"""
Constants and functions to create the Fashion MNIST dataset and dataloaders.

The original dataset contains 10 different labels:
  - `0`: T-shirt/top
  - `1`: Trouser
  - `2`: Pullover
  - `3`: Dress
  - `4`: Coat
  - `5`: Sandal
  - `6`: Shirt
  - `7`: Sneaker
  - `8`: Bag
  - `9`: Ankle boot

In this case, we want you to group the original labels
in 5 new labels with the following mapping between themmapping between them:
  - `Upper part`: T-shirt/top + Pullover + Coat + Shirt
  - `Bottom part`: Trouser
  - `One piece`: Dress
  - `Footwear`: Sandal + Sneaker + Ankle boot
  - `Bags`: Bag

Author: Mario Parreño Lara (maparla@proton.me)

Date: 07/10/2023
"""

### Import Libraries
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import pytorch_lightning as pl
import albumentations as A
from etl.normalization import apply_normalization


############################################################################
########################## Define constants ################################
############################################################################

# Define a mapping from original labels to new labels
LABEL_TO_CLASS = {
    0: 'Upper part',
    1: 'Bottom part',
    2: 'One piece',
    3: 'Footwear',
    4: 'Bags',
}

LABEL_MAPPING = {
    0: 0,  # T-shirt/top  -> 'Upper part'
    1: 1,  # Trouser      -> 'Bottom part'
    2: 0,  # Pullover     -> 'Upper part'
    3: 2,  # Dress        -> 'One piece'
    4: 0,  # Coat         -> 'Upper part'
    5: 3,  # Sandal       -> 'Footwear' 
    6: 0,  # Shirt        -> 'Upper part'
    7: 3,  # Sneaker      -> 'Footwear'
    8: 4,  # Bag          -> 'Bags'
    9: 3   # Ankle boot   -> 'Footwear' 
}

IN_CHANNELS = 1  # Grayscale
NUM_CLASSES = len(LABEL_TO_CLASS)
CLASS_NAMES = [LABEL_TO_CLASS[i] for i in range(NUM_CLASSES)]

############################################################################
######################### Create the dataset ###############################
############################################################################

class CustomFashionMNIST(datasets.FashionMNIST):
    """Custom Fashion MNIST dataset

    Attributes:
        label_mapping (dict<int, int>): Mapping from original labels to new labels
        augmentation (A.Compose): Augmentation pipeline
        normalization (str): Normalization method
    """
    def __init__(
            self,
            root: str,
            train: bool = True,
            augmentation = None,
            normalization: str = "255",
            download: bool = True,
        ):
        super(CustomFashionMNIST, self).__init__(
            root=root,
            train=train,
            download=download
        )
        self.label_mapping = LABEL_MAPPING
        self.augmentation = augmentation
        self.normalization = normalization

    def __getitem__(self, index):
        img, target = super(CustomFashionMNIST, self).__getitem__(index)
        
        if self.augmentation is not None:
            img = self.augmentation(image=np.array(img))["image"]

        img = apply_normalization(
            image=np.array(img),
            normalization_type=self.normalization
        )

        # Convert img to float tensor and add channel dimension
        img = torch.from_numpy(img).float().unsqueeze(0)

        new_target = self.label_mapping[target]
        return img, new_target


class CustomFashionMNISTDataModule(pl.LightningDataModule):
    """Lightning DataModule for the Fashion MNIST dataset

    Attributes:
        data_root (str): Path to the Fashion MNIST dataset
        train_augmentation (A.Compose): Augmentation pipeline for training
        val_augmentation (A.Compose): Augmentation pipeline for validation
        normalization (str): Normalization method
        batch_size (int): Batch size
        val_size (float): Percentage of samples to use for validation
        random_seed (int): Random seed for reproducibility
        num_workers (int): Number of workers for the dataloaders
    """

    def __init__(
            self,
            train_augmentation: A.Compose,
            val_augmentation: A.Compose,
            normalization: str,
            batch_size: int = 64,
            val_size: float = 0.15,
            random_seed: int = 42,
            num_workers: int = 0,
        ):
        """Inits the CustomFashionMNISTDataModule class"""
        super().__init__()
        self.data_root = os.environ.get("FASHION_DIR")
        if self.data_root is None:
            raise ValueError("The system variable 'FASHION_DIR' is required.")
            
        self.train_augmentation = train_augmentation
        self.val_augmentation = val_augmentation
        self.normalization = normalization
        self.batch_size = batch_size
        self.val_size = val_size
        self.random_seed = random_seed
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        """Creates the train, val and test datasets"""
        full_train_ds = CustomFashionMNIST(
            root=self.data_root, train=True,
            augmentation=self.train_augmentation,
            normalization=self.normalization
        )

        # Calculate the number of validation samples
        num_train = len(full_train_ds)
        num_val = int(self.val_size * num_train)

        # Set the random seed for reproducibility
        random.seed(self.random_seed)

        # Split the dataset into train and val
        indices = list(range(num_train))
        random.shuffle(indices)
        train_indices, val_indices = indices[num_val:], indices[:num_val]

        # Create train and val datasets
        self.train_ds = torch.utils.data.Subset(full_train_ds, train_indices)
        self.val_ds = torch.utils.data.Subset(full_train_ds, val_indices)

        self.test_ds = CustomFashionMNIST(
            root=self.data_root, train=False,
            augmentation=self.val_augmentation,
            normalization=self.normalization
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
