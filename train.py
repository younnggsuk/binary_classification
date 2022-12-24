import os
import sys
import timm
import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_accuracy,
    binary_recall,
    binary_specificity,
    binary_precision,
    binary_confusion_matrix,
)

import utils
from data import get_dataloader
from logger import get_logger
from models.lr_scheduler import LinearWarmupCosineAnnealingLR


def train_one_epoch(model, loss_fn, optimizer, data_loader):
    """
    Train model for one epoch.
    """
    running_loss = 0.0
    
    for images, true_labels, _ in tqdm(data_loader, total=len(data_loader), desc="Train"):
        images = images.cuda(non_blocking=True)
        true_labels = true_labels.float().cuda(non_blocking=True)
        
        pred_logits = model(images)
        loss = loss_fn(pred_logits, true_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = utils.reduce_tensor(loss)
        running_loss += loss.item()
    
    running_loss /= len(data_loader)
    
    return running_loss


def eval_one_epoch(model, loss_fn, data_loader):
    """
    Evaluate model for one epoch.
    """
    running_loss = 0.0
    result_dict = {
        "pred_probs":  [],
        "true_labels": [],
    }

    for images, true_labels, image_paths in tqdm(data_loader, total=len(data_loader), desc="Valid"):
        images = images.cuda(non_blocking=True)
        true_labels = true_labels.float().cuda(non_blocking=True)
        
        with torch.no_grad():
            pred_logits = model(images)
        
        loss = loss_fn(pred_logits, true_labels)
        loss = utils.reduce_tensor(loss)
        running_loss += loss.item()
        
        pred_probs = torch.softmax(pred_logits, dim=1)
        result_dict["pred_probs"].append(torch.cat(utils.gather_tensor(pred_probs)).cpu())
        result_dict["true_labels"].append(torch.argmax(torch.cat(utils.gather_tensor(true_labels)), dim=1).cpu())
    
    running_loss /= len(data_loader)
    pred_probs  = torch.cat([result_dict_batch for result_dict_batch in result_dict["pred_probs"]] , dim=0)
    true_labels = torch.cat([result_dict_batch for result_dict_batch in result_dict["true_labels"]], dim=0)
    
    return running_loss, pred_probs, true_labels


def log_one_epoch(epoch, result_dict, tb_writer, log_dir_cm, pred_probs, true_labels):
    """
    Log metrics and confusion matrices for one epoch.
    """
    result_dict.update({
        "auroc": binary_auroc(pred_probs[:, 1], true_labels),
        "acc_0.1": binary_accuracy(pred_probs[:, 1], true_labels, threshold=0.1),
        "acc_0.2": binary_accuracy(pred_probs[:, 1], true_labels, threshold=0.2),
        "acc_0.3": binary_accuracy(pred_probs[:, 1], true_labels, threshold=0.3),
        "acc_0.4": binary_accuracy(pred_probs[:, 1], true_labels, threshold=0.4),
        "acc_0.5": binary_accuracy(pred_probs[:, 1], true_labels, threshold=0.5),
        "acc_0.6": binary_accuracy(pred_probs[:, 1], true_labels, threshold=0.6),
        "acc_0.7": binary_accuracy(pred_probs[:, 1], true_labels, threshold=0.7),
        "acc_0.8": binary_accuracy(pred_probs[:, 1], true_labels, threshold=0.8),
        "acc_0.9": binary_accuracy(pred_probs[:, 1], true_labels, threshold=0.9),
        "sensitivity_0.1": binary_recall(pred_probs[:, 1], true_labels, threshold=0.1),
        "sensitivity_0.2": binary_recall(pred_probs[:, 1], true_labels, threshold=0.2),
        "sensitivity_0.3": binary_recall(pred_probs[:, 1], true_labels, threshold=0.3),
        "sensitivity_0.4": binary_recall(pred_probs[:, 1], true_labels, threshold=0.4),
        "sensitivity_0.5": binary_recall(pred_probs[:, 1], true_labels, threshold=0.5),
        "sensitivity_0.6": binary_recall(pred_probs[:, 1], true_labels, threshold=0.6),
        "sensitivity_0.7": binary_recall(pred_probs[:, 1], true_labels, threshold=0.7),
        "sensitivity_0.8": binary_recall(pred_probs[:, 1], true_labels, threshold=0.8),
        "sensitivity_0.9": binary_recall(pred_probs[:, 1], true_labels, threshold=0.9),
        "specificity_0.1": binary_specificity(pred_probs[:, 1], true_labels, threshold=0.1),
        "specificity_0.2": binary_specificity(pred_probs[:, 1], true_labels, threshold=0.2),
        "specificity_0.3": binary_specificity(pred_probs[:, 1], true_labels, threshold=0.3),
        "specificity_0.4": binary_specificity(pred_probs[:, 1], true_labels, threshold=0.4),
        "specificity_0.5": binary_specificity(pred_probs[:, 1], true_labels, threshold=0.5), 
        "specificity_0.6": binary_specificity(pred_probs[:, 1], true_labels, threshold=0.6),
        "specificity_0.7": binary_specificity(pred_probs[:, 1], true_labels, threshold=0.7),
        "specificity_0.8": binary_specificity(pred_probs[:, 1], true_labels, threshold=0.8),
        "specificity_0.9": binary_specificity(pred_probs[:, 1], true_labels, threshold=0.9),
        "precision_0.1": binary_precision(pred_probs[:, 1], true_labels, threshold=0.1),
        "precision_0.2": binary_precision(pred_probs[:, 1], true_labels, threshold=0.2),
        "precision_0.3": binary_precision(pred_probs[:, 1], true_labels, threshold=0.3),
        "precision_0.4": binary_precision(pred_probs[:, 1], true_labels, threshold=0.4),
        "precision_0.5": binary_precision(pred_probs[:, 1], true_labels, threshold=0.5), 
        "precision_0.6": binary_precision(pred_probs[:, 1], true_labels, threshold=0.6),
        "precision_0.7": binary_precision(pred_probs[:, 1], true_labels, threshold=0.7),
        "precision_0.8": binary_precision(pred_probs[:, 1], true_labels, threshold=0.8),
        "precision_0.9": binary_precision(pred_probs[:, 1], true_labels, threshold=0.9),
    })

    for key, value in result_dict.items():
        tb_writer.add_scalar(key, value, epoch)
    
    log_dir_cm = os.path.join(log_dir_cm, f"{epoch:03d}")
    if not os.path.exists(log_dir_cm):
        os.makedirs(log_dir_cm)
        
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        axis_str = ["0", "1"]
        df_cm = pd.DataFrame(
            binary_confusion_matrix(
                pred_probs[:, 1], 
                true_labels,
                threshold=threshold
            ).int().numpy(),
            index=axis_str,
            columns=axis_str,
        )
        fig = plt.figure(figsize = (8, 5))
        sns.heatmap(
            df_cm,
            annot=True, 
            cmap="Blues",
            cbar="False",
            fmt='g',
        )
        sensitivity = result_dict[f"sensitivity_{threshold}"]
        specificity = result_dict[f"specificity_{threshold}"]
        plt.xlabel("Pred")
        plt.ylabel("True")
        plt.title(f"[Threshold: {threshold}] sensitivity: {sensitivity:.4f} / specificity: {specificity:.4f}")
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir_cm, f'confusion_matrix_threshold{threshold}.png'))
        plt.close(fig)

    return result_dict


def get_current_lr(optimizer):
    """
    Get current learning rate from optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def main():
    # parse args
    args = utils.get_train_args_parser().parse_args()
        
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print("Should be launched with torch.distributed.launch or torchrun")
        sys.exit(1)

    # init distributed training
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    dist.barrier()
    utils.setup_for_distributed(args.rank == 0)
    cudnn.benchmark = True
        
    # fix seed
    utils.fix_random_seeds(args.seed)
    
    # experiment dirs
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, args.exp_name)
    log_dir_cm = os.path.join(log_dir, "confusion_matrix")
    log_dir_csv = os.path.join(log_dir, "log_csv")
    log_dir_ckpt = os.path.join(log_dir, "checkpoints")

    if utils.is_main_process():
        for dir_path in [log_dir,
                         log_dir_cm,
                         log_dir_csv,
                         log_dir_ckpt]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # tensorboard setting
        tb_writer = SummaryWriter(os.path.join(log_dir, f"tensorboard"))
        
    # get logger
    logger = get_logger(log_dir, resume="False", is_rank0=utils.is_main_process())
        
    # print args
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
        
    # load data list
    data_list_dict = utils.load_data_list_from_json(args.data_list_json)
    train_list = data_list_dict["train_list"]
    val_list = data_list_dict["val_list"]
    random.shuffle(train_list)
    logger.info(f"Data size: train {len(train_list)}, val {len(val_list)}")
    
    # data loaders
    train_loader = get_dataloader(
        root_dir=args.root_dir,
        data_list=train_list,
        crop_size=args.crop_size,
        batch_size_per_gpu=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        is_train=True
    )
    val_loader = get_dataloader(
        root_dir=args.root_dir,
        data_list=val_list,
        crop_size=args.crop_size,
        batch_size_per_gpu=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        is_train=False
    )

    # build model
    model = timm.create_model(
        args.arch, 
        in_chans=args.in_chans, 
        num_classes=args.num_classes,
        pretrained=args.pretrained, 
    )
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # optimizer
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Optimizer not supported")
    
    # lr scheduler
    if args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.lr_scheduler == "cosine_anneling":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    elif args.lr_scheduler == "linear_warmup_cosine_anneling":
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, 
                                                  warmup_epochs=5,
                                                  max_epochs=args.max_epochs)
    else:
        raise ValueError("LR Scheduler not supported")

    # resume from checkpoint (if exists)
    run_variables = {"epoch": 0,
                     "best_val_loss": np.inf,
                     "best_save_path": None}
    utils.resume_from_checkpoint(
        ckpt_path=os.path.join(log_dir_ckpt, "checkpoint_last.pth"),
        logger=logger,
        run_variables=run_variables,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    
    # training variables
    best_val_loss = run_variables["best_val_loss"]
    best_save_path = run_variables["best_save_path"]
    epoch_start = run_variables["epoch"]
    epoch_end = args.max_epochs
    if args.early_stop is not None:
        epoch_end = args.early_stop
    
    # start training
    logger.info("\nStart training!!")
    for epoch in range(epoch_start, epoch_end):
        logger.info(f"Epoch: [{epoch:03d}/{epoch_end - 1:03d}]")
        
        # train model
        model.train()
        train_epoch_loss = train_one_epoch(model, loss_fn, optimizer, train_loader)
        scheduler.step()
        
        # eval model
        model.eval()
        val_epoch_loss, pred_probs, true_labels = eval_one_epoch(model, loss_fn, val_loader)
        
        # print loss
        logger.info(f"train loss: {train_epoch_loss:.4f}, val loss: {val_epoch_loss:.4f}")
        
        if utils.is_main_process():
            # log metrics
            result_dict = {
                "train_loss": train_epoch_loss,
                "val_loss": val_epoch_loss,
                "learning_rate": get_current_lr(optimizer)
            }
            result_dict = log_one_epoch(
                epoch, 
                result_dict,
                tb_writer,
                log_dir_cm, 
                pred_probs, 
                true_labels, 
            )
            pd.DataFrame(result_dict, index=[0]).to_csv(os.path.join(log_dir_csv, f"epoch_{epoch:03d}_result.csv"), index=False)
            
            # checkpoint dict
            save_dict = {
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'best_save_path': best_save_path,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            
            # save best checkpoint
            if args.save_best is not None:
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    if best_save_path is not None:
                        os.remove(best_save_path)
                    best_save_path = os.path.join(log_dir_ckpt, f"checkpoint_best.pth")
                    
                    save_dict["best_val_loss"] = best_val_loss
                    save_dict["best_save_path"] = best_save_path
                    torch.save(save_dict, best_save_path)
                    logger.info(f"Saved best model at {best_save_path}")
            
            # save checkpoint every saveckp_freq epochs
            if (epoch + 1) % args.saveckp_freq == 0:
                freq_save_path = os.path.join(log_dir_ckpt, f"checkpoint{epoch:03d}.pth")
                torch.save(save_dict, freq_save_path)
                logger.info(f"Saved model at {freq_save_path}")

            # save last checkpoint to resume training
            last_save_path = os.path.join(log_dir_ckpt, "checkpoint_last.pth")
            torch.save(save_dict, last_save_path)
            
    logger.info("Done!!\n")
    

if __name__ == '__main__':
    main()
