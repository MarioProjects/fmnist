"""
Training script for the Fashion MNIST classification task.

Author: Mario Parreño Lara (maparla@proton.me)

Date: 07/10/2023
"""
import wandb

from pytorch_lightning import Trainer

import constants
from arguments import config
from core import FashionModule, log_model
from etl.dataset import CustomFashionMNISTDataModule
from etl.augmentations import get_transforms

##################################################
################## Main ##########################
##################################################

# Initialize wandb run and download artifacts
run = wandb.init(
    project=constants.WANDB_PROJECT,
    entity=constants.ENTITY,
    config=config,
    job_type="training"
)

train_transforms, test_transforms = get_transforms(config)
dm = CustomFashionMNISTDataModule(
    train_transforms, test_transforms, config.normalization,
    config.batch_size, config.val_size, config.seed, config.num_workers
)
dm.setup()

model = FashionModule(config)

trainer = Trainer(
    max_epochs=config.epochs,
    enable_checkpointing=False,  # We will save the model manually
    logger=False  # We are using wandb manually as logger
)

trainer.fit(model, dm)

if config.save_dir is not None:
    log_model(trainer, config)

wandb.finish()
