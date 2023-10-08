"""
Testing script for the Fashion MNIST classification task.

Author: Mario Parreño Lara (maparla@proton.me)

Date: 07/10/2023
"""
import wandb

from pytorch_lightning import Trainer

import constants
from arguments import config
from core import FashionModule
from etl.dataset import CustomFashionMNISTDataModule
from etl.augmentations import get_transforms

##################################################
################## Main ##########################
##################################################

if config.checkpoint is None:
    raise ValueError("Checkpoint path is not specified")

# Initialize wandb run and download artifacts
run = wandb.init(
    project=constants.WANDB_PROJECT,
    entity=constants.ENTITY,
    config=config,
    job_type="testing"
)

train_transforms, test_transforms = get_transforms(config)
dm = CustomFashionMNISTDataModule(
    train_transforms, test_transforms, config.normalization,
    config.batch_size, random_seed=config.seed,
    num_workers=config.num_workers
)
dm.setup()

model = FashionModule.load_from_checkpoint(config.checkpoint)

trainer = Trainer(
    max_epochs=config.epochs,
    logger=False  # We are using wandb "manually" as logger
)

trainer.test(model, datamodule=dm)

wandb.finish()
