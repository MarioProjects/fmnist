"""
Core module for the Fashion Module Model.

Author: Mario Parreño Lara (maparla@proton.me)

Date: 07/10/2023
"""
import os
from argparse import Namespace
import wandb

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

import torchmetrics

from models.hub import get_model
from models.losses import get_criterion
from models.optimizers import get_optimizer
from models.schedulers import get_scheduler

from etl.dataset import LABEL_TO_CLASS, NUM_CLASSES, CLASS_NAMES


def log_model(trainer: pl.Trainer, config: Namespace) -> None:
    """Logs the model to wandb and saves it locally

    Args:
        trainer (pytorch_lightning.Trainer): The trainer
        config (Namespace): The config object

    Returns:
        None
    """

    last_model_path = os.path.join(config.save_dir, "last_model.ckpt")
    trainer.save_checkpoint(last_model_path)
    last_model_artifact = wandb.Artifact('last_model', type='model')
    last_model_artifact.add_file(last_model_path)
    wandb.log_artifact(last_model_artifact, aliases=["last"])

    if config.val_size > 0:
        best_model_artifact = wandb.Artifact('best_model', type='model')
        best_model_path = os.path.join(config.save_dir, "best_model.ckpt")
        best_model_artifact.add_file(best_model_path)
        wandb.log_artifact(best_model_artifact, aliases=["best_model"])

#################################################
## Create the trainer -> train the model & log ##
#################################################

class FashionModule(LightningModule):
    """The Fashion Module Model

    Args:
        config (Namespace): The config object

    Attributes:
        save_dir (str): The directory to save the model
        lr (float): The learning rate
        criterion (torch.nn.Module): The criterion to optimize
        model (torch.nn.Module): The model to train
        optimizer (str): The optimizer to use
        scheduler (str): The scheduler to use
        milestones (list): The milestones for the scheduler
        step_gamma (float): The step gamma for the scheduler
        train_loss (torchmetrics.MeanMetric): The train loss
        train_accuracy (torchmetrics.Accuracy): The train accuracy
        train_f1 (torchmetrics.F1Score): The train f1 score
        val_accuracy (torchmetrics.Accuracy): The validation accuracy
        val_f1 (torchmetrics.F1Score): The validation f1 score
        test_accuracy (torchmetrics.Accuracy): The test accuracy
        test_f1 (torchmetrics.F1Score): The test f1 score
        test_summary_table (wandb.Table): The test summary table
        test_ground_truth (list): The test ground truth
        test_predictions (list): The test predictions
        best_val_acc (float): The best validation accuracy
        save_hyperparameters (function): The function to save the hyperparameters
        forward (function): The forward function
        training_step (function): The training step function
        on_train_epoch_end (function): The training epoch end function
        validation_step (function): The validation step function
        on_validation_epoch_end (function): The validation epoch end function
        test_step (function): The test step function
        on_test_epoch_end (function): The test epoch end function
        configure_optimizers (function): The function to configure the optimizers
        
    Returns:
        None
    """
    def __init__(self, config):
        super().__init__()

        self.save_dir = config.save_dir
        self.lr = config.lr

        self.criterion = get_criterion(config)
        self.model = get_model(config)

        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        self.milestones = config.milestones
        self.step_gamma = config.step_gamma

        self.train_loss = torchmetrics.MeanMetric()
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=NUM_CLASSES
        )
        self.train_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=NUM_CLASSES, average="weighted"
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=NUM_CLASSES
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=NUM_CLASSES, average="weighted"
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=NUM_CLASSES
        )
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=NUM_CLASSES, average="weighted"
        )
        confidence_cols = [f"Confidence {c}" for c in CLASS_NAMES]
        self.test_summary_table = wandb.Table(
            columns=["Images", "Ground Truth", "Prediction"] + confidence_cols
        )
        self.test_ground_truth = []
        self.test_predictions = []
        self.best_val_acc = 0
        self.save_hyperparameters()

    def forward(self, x):
        """Used when the model is called."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Called at the end of the training step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_loss(loss)

        # Compute accuracy with logits and labels
        self.train_accuracy(logits, y)
        self.train_f1(logits, y)

        return loss

    def on_train_epoch_end(self):
        """Called at the end of the training epoch."""
        train_epoch_loss = self.train_loss.compute()
        train_epoch_accuracy = self.train_accuracy.compute()
        train_epoch_f1 = self.train_f1.compute()
        wandb.log(
            {"metrics/train_accuracy": train_epoch_accuracy}, step=self.current_epoch
        )
        wandb.log(
            {"metrics/train_loss": train_epoch_loss}, step=self.current_epoch
        )
        wandb.log(
            {"metrics/train_f1": train_epoch_f1}, step=self.current_epoch
        )
        wandb.log(
            {"metrics/lr": self.trainer.optimizers[0].param_groups[0]['lr']},
            step=self.current_epoch
        )
        self.log("train_f1", train_epoch_f1, prog_bar=True)
        self.log("train_loss", train_epoch_loss, prog_bar=True)
        self.train_accuracy.reset()
        self.train_loss.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        """Called at the end of the validation step."""
        x, y = batch
        logits = self(x)
        
        # Compute accuracy with logits and labels
        self.val_accuracy(logits, y)
        self.val_f1(logits, y)
    
    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch."""
        val_epoch_accuracy = self.val_accuracy.compute()
        val_epoch_f1 = self.val_f1.compute()
        wandb.log({"metrics/val_accuracy": val_epoch_accuracy}, step=self.current_epoch)
        wandb.log({"metrics/val_f1": val_epoch_f1}, step=self.current_epoch)
        self.log("val_f1", val_epoch_f1, prog_bar=True)
        
        if self.save_dir is not None and val_epoch_accuracy > self.best_val_acc:
            self.best_val_acc = val_epoch_accuracy
            save_path = os.path.join(self.save_dir, "best_model.ckpt")
            self.trainer.save_checkpoint(save_path)

        self.val_accuracy.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        """Called at the end of the test step."""
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        for i in range(len(x)):
            image = x[i].cpu().permute(1, 2, 0).numpy().squeeze()
            class_probs = torch.softmax(logits[i], dim=0).cpu().numpy()
            self.test_summary_table.add_data(
                wandb.Image(image),
                LABEL_TO_CLASS[y[i].item()],
                LABEL_TO_CLASS[preds[i].item()],
                *class_probs
            )
            
        self.test_ground_truth.extend(y.cpu().numpy())
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_accuracy(logits, y)
        self.test_f1(logits, y)
    
    def on_test_epoch_end(self):
        """Called at the end of the test epoch."""
        test_epoch_accuracy = self.test_accuracy.compute()
        test_epoch_f1 = self.test_f1.compute()
        wandb.summary["test_accuracy"] = test_epoch_accuracy
        wandb.summary["test_f1"] = test_epoch_f1

        wandb.log({"metrics/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=self.test_ground_truth,
            preds=self.test_predictions,
            class_names=CLASS_NAMES
        )})

        wandb.log({"metrics/test_summary_table": self.test_summary_table})
        self.test_accuracy.reset()
        self.test_f1.reset()

    def configure_optimizers(self):
        """Configures the optimizers."""
        optimizer = get_optimizer(self)
        scheduler = get_scheduler(self, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
