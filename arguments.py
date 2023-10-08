"""
This file contains all the arguments for the pipeline.

Author: Mario Parreño Lara (maparla@proton.me)

Date: 07/10/2023
"""
import os
import uuid
import argparse


class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


parser = argparse.ArgumentParser(
    description='Fashion MNIST Pipeline', formatter_class=SmartFormatter

)

parser.add_argument(
    '--save_dir', type=str, default=None,
    help='R|Directory for saving artifacts and checkpoints'
)

parser.add_argument(
    '--model', type=str, default='resnet18', help='R|Model'
)

parser.add_argument(
    '--normalization', type=str, default='255',
    help='R|Data normalization method'
)

parser.add_argument(
    '--checkpoint', type=str, default=None, help='R|Checkpoint path'
)

parser.add_argument(
    '--lr', type=float, default=0.01, help='R|Learning rate'
)

parser.add_argument(
    '--batch_size', type=int, default=8, help='R|Batch size'
)

parser.add_argument(
    '--epochs', type=int, default=15, help='R|Number of epochs'
)

parser.add_argument(
    '--optimizer', type=str, default='SGD', help='R|Optimizer for training'
)

parser.add_argument(
    '--scheduler', type=str, default='MultiStepLR', help='R|Scheduler for optimizer'
)

parser.add_argument(
    '--milestones', type=str,
    default="5-10", help='R|Milestones for scheduler'
)

parser.add_argument(
    '--step_gamma', type=float, default=0.1, help='R|Gamma for scheduler'
)

parser.add_argument(
    '--weights', type=str, default='random', help='R|Model weights initialization'
)

parser.add_argument(
    '--num_workers', type=int, default=4, help='R|Number of workers for dataloader'
)

parser.add_argument(
    '--loss', type=str, default='cross_entropy', help='R|Loss'
)

parser.add_argument(
    '--val_size', type=float, default=0.2,
    help='R|Validation split fraction for training'
)

parser.add_argument(
    '--seed', type=int, default=42,
    help='R|Seed for reproducibility. Used in validation split'
)

#############################################################################
###################### DATA AUGMENTATION ARGUMENTS ##########################
#############################################################################

parser.add_argument(
    '--p_rotate', type=float, default=0.5, help='R|Probability of rotation'
)

parser.add_argument(
    '--p_posterize', type=float, default=0.5, help='R|Probability of posterize'
)

parser.add_argument(
    '--p_shift', type=float, default=0.5, help='R|Probability of shift'
)

parser.add_argument(
    '--p_contrast', type=float, default=0.5, help='R|Probability of contrast'
)

parser.add_argument(
    '--p_brightness', type=float, default=0.5, help='R|Probability of brightness'
)

parser.add_argument(
    '--p_hflip', type=float, default=0.5, help='R|Probability of horizontal flip'
)

parser.add_argument(
    '--p_vflip', type=float, default=0.5, help='R|Probability of vertical flip'
)

parser.add_argument(
    '--p_random_crop', type=float, default=0.5, help='R|Probability of random crop'
)

parser.add_argument(
    '--p_blur', type=float, default=0.5, help='R|Probability of blur'
)

parser.add_argument(
    '--p_dropout', type=float, default=0.5, help='R|Probability of dropout'
)

config = parser.parse_args()

if config.save_dir is not None and config.save_dir != "":
    config.save_dir = os.path.join(config.save_dir, str(uuid.uuid4()))
    os.makedirs(config.save_dir, exist_ok=True)
    print(f"\nSaving artifacts to {config.save_dir}\n")

# Convert milestones to list
config.milestones = [int(i) for i in config.milestones.split('-')]