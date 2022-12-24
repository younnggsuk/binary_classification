import os
import json
import random
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist


def fix_random_seeds(seed=1111):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    

def is_dist_avail_and_initialized():
    """
    Check if distributed training is available and initialized.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    Get world size.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Get rank.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    Check if current process is main process.
    """
    return get_rank() == 0


def reduce_tensor(tensor):
    """
    Reduce distributed tensor.
    """
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
    reduced_tensor /= dist.get_world_size()
    return reduced_tensor


def gather_tensor(tensor):
    """
    Gather distributed tensor.
    """
    gathered_tensor = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_tensor, tensor)
    return gathered_tensor
    
    
def setup_for_distributed(is_master):
    """
    Disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
    
def load_data_list_from_json(json_path):
    """
    Load data list from json file.
    """
    with open(json_path, "r") as f:
        data_list = json.load(f)
    return data_list


def load_pretrained_model_weights(model, ckpt_path, logger=None):
    """
    Load pretrained model weights from checkpoint.
    """
    
    # if logger is not None, print to logger
    if logger:
        print = logger.info
    
    if os.path.isfile(ckpt_path):
        print(f'Pretrained weights found at {ckpt_path}')
        model_state_dict = torch.load(ckpt_path, map_location="cpu")["model"]
        msg = model.load_state_dict(model_state_dict, strict=False)
        print(f"Loaded pretrained weights from {ckpt_path} with msg: {msg}")
    else:
        print("No pretrained weights found.")
    
    
def resume_from_checkpoint(ckpt_path, logger=None, run_variables=None, **kwargs):
    """
    Resume training from checkpoint.
    """
    # if checkpoint is not found, start from scratch
    if not os.path.isfile(ckpt_path):
        return
    
    # if logger is not None, print to logger
    if logger:
        print = logger.info
    
    # if checkpoint is found, load state dict from checkpoint
    print(f"Found checkpoint at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(f"Loaded '{key}' from checkpoint with msg {msg}")
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
                print(f"Loaded '{key}' from checkpoint")
        else:
            print(f"Key '{key}' not found in checkpoint: {ckpt_path}")
            
    if run_variables is not None:
        for key in run_variables:
            if key in checkpoint:
                run_variables[key] = checkpoint[key]


def get_train_args_parser():
    """
    Get train arguments parser.
    """
    parser = argparse.ArgumentParser(description='Classification_Train', add_help=True)

    # Model parameters
    parser.add_argument(
        '--arch', default='resnet50', type=str,
        choices=['resnet34', 'resnet50'],
        help="""Name of architecture to train."""
    )
    parser.add_argument(
        '--in_chans', default=3, type=int, 
        help="""Number of input channels - default 3 (for RGB images)."""
    )
    parser.add_argument(
        '--num_classes', default=2, type=int,
        help="""Number of classes to classify - default 2 (for binary classification)."""
    )
    parser.add_argument(
        '--pretrained', default=True, type=bool,
        help="""Whether to use pretrained weights - default True."""
    )

    # Training parameters
    parser.add_argument(
        '--max_epochs', default=100, type=int,
        help="""Number of total epochs to run - default 100."""
    )
    parser.add_argument(
        '--early_stop', default=None, type=int,
        help="""Number of epochs to early stop."""
    )
    
    # Optimizer parameters
    parser.add_argument(
        '--optimizer', default='adamw', type=str,
        choices=['sgd', 'adam', 'adamw'], 
        help="""Type of optimizer."""
    )
    parser.add_argument(
        '--lr', default=0.001, type=float, 
        help="""Initial learning rate - default 0.001."""
    )
    parser.add_argument(
        '--lr_scheduler', default='linear_warmup_cosine_anneling', type=str,
        choices=['step', 'cosine_anneling', 'linear_warmup_cosine_anneling'],
        help="""Type of learning rate scheduler."""
    )

    # Data parameters
    parser.add_argument(
        '--root_dir', default='/path/to/data/train/', type=str,
        help="""Path to the root directory of the dataset."""
    )
    parser.add_argument(
        '--data_list_json', default='/path/to/data_list.json', type=str,
        help="""Path to the json file containing the list of train/val images."""
    )
    parser.add_argument(
        '--crop_size', default=224, type=int,
        help="""Size of the image crop (default: 224)."""
    )
    parser.add_argument(
        '--batch_size_per_gpu', default=64, type=int,
        help="""Per-GPU batch-size : number of distinct images loaded on one GPU."""
    )
    parser.add_argument(
        '--num_workers', default=8, type=int,
        help="""Number of workers for the data loader."""
    )
    
    # Misc
    parser.add_argument(
        '--exp_name', default=None, type=str,
        help="""Name of the experiment."""
    )
    parser.add_argument(
        '--log_dir', default="./logs", type=str,
        help="""Directory to save test results."""
    )
    parser.add_argument(
        '--save_best', default=True, type=bool,
        help="""Whether to save the best model."""
    )
    parser.add_argument(
        '--saveckp_freq', default=10, type=int,
        help="""Frequency of saving checkpoints."""
    )
    parser.add_argument(
        '--seed', default=1111, type=int,
        help="""Random seed."""
    )
    parser.add_argument(
        '--dist_url', default="env://", type=str,
        help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html"""
    )
    parser.add_argument(
        '--local_rank', default=0, type=int,
        help="Please ignore and do not set this argument."
    )
    return parser


def get_test_args_parser():
    """
    Get test arguments parser.
    """
    parser = argparse.ArgumentParser(description='Classification_Test', add_help=True)

    # Model parameters
    parser.add_argument(
        '--arch', default='resnet50', type=str,
        choices=['resnet34', 'resnet50'],
        help="""Name of architecture to train."""
    )
    parser.add_argument(
        '--in_chans', default=3, type=int, 
        help="""Number of input channels - default 3 (for RGB images)."""
    )
    parser.add_argument(
        '--num_classes', default=2, type=int,
        help="""Number of classes to classify - default 2."""
    )
    parser.add_argument(
        '--ckpt_path', default="/path/to/checkpoint.pth", type=str,
        help="""Path to the checkpoint file."""
    )
    parser.add_argument(
        '--threshold', default=None, type=float,
        help="""Classification threshold (binary classification only) - default None."""
    )

    # Data parameters
    parser.add_argument(
        '--root_dir', default='/path/to/data/test/', type=str,
        help="""Path to the root directory of the dataset."""
    )
    parser.add_argument(
        '--data_list_json', default='/path/to/data_list.json', type=str,
        help="""Path to the json file containing the list of test images."""
    )
    parser.add_argument(
        '--crop_size', default=224, type=int,
        help="""Size of the image crop (default: 224)."""
    )
    parser.add_argument(
        '--batch_size_per_gpu', default=64, type=int,
        help="""Per-GPU batch-size : number of datas to be loaded on one GPU."""
    )
    parser.add_argument(
        '--num_workers', default=8, type=int,
        help="""Number of workers for the data loader."""
    )
    
    # Misc
    parser.add_argument(
        '--exp_name', default=None, type=str,
        help="""Name of the experiment."""
    )
    parser.add_argument(
        '--result_dir', default="./results", type=str,
        help="""Path to the directory where the results will be saved."""
    )
    parser.add_argument(
        '--seed', default=1111, type=int,
        help="""Random seed."""
    )
    parser.add_argument(
        '--dist_url', default="env://", type=str,
        help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html"""
    )
    parser.add_argument(
        '--local_rank', default=0, type=int,
        help="Please ignore and do not set this argument."
    )
    return parser
