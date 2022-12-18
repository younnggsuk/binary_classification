import os
import cv2
import sys
import timm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
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
from models.grad_cam import CAM


def eval_one_epoch_with_save_cam(cam, data_loader, result_dir_cam_correct, result_dir_cam_incorrect, threshold=0.5):
    result_dict = {
        "pred_probs": [],
        "true_labels": [],
    }
    
    for images, true_labels, image_paths in tqdm(data_loader, total=len(data_loader), desc="Test"):
        images = images.cuda(non_blocking=True)
        true_labels = true_labels.cuda(non_blocking=True)
        
        gray_cams = cam(input_tensor=images)
        pred_logits = cam.outputs
        pred_probs = torch.softmax(pred_logits, dim=1)
        
        for image_path_, image_, pred_prob_, true_label_, gray_cam_ in zip(image_paths, images, pred_probs, true_labels, gray_cams):
            original_image = Image.open(image_path_).convert('RGB')
            
            denorm_image = image_.cpu().numpy().transpose(1, 2, 0) * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
            denorm_image = (denorm_image * 255.).astype("uint8")
            
            gray_cam_ = cv2.applyColorMap((gray_cam_ * 255.).astype("uint8"), cv2.COLORMAP_JET)
            image_cam = cv2.addWeighted(denorm_image, 0.5, cv2.cvtColor(gray_cam_, cv2.COLOR_BGR2RGB), 0.5, 0)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            pred_prob_ = pred_prob_.cpu().detach().numpy()
            true_label_ = true_label_.cpu().detach().numpy()
            plt.suptitle(
                f"Pred: [{pred_prob_[0]:.3f}, {pred_prob_[1]:.3f}] / True: {true_label_}",
                fontsize=20
            )
            
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            
            axes[1].imshow(denorm_image)
            axes[1].set_title("Crop Image")
            axes[1].axis("off")
            
            axes[2].imshow(image_cam)
            axes[2].set_title("CAM")
            axes[2].axis("off")
            
            plt.tight_layout()
            
            # correctness
            if int(pred_prob_[1] >= threshold) == np.argmax(true_label_):
                path = os.path.join(result_dir_cam_correct, os.path.basename(image_path_))
            else:
                path = os.path.join(result_dir_cam_incorrect, os.path.basename(image_path_))
            plt.savefig(path)
            plt.close(fig)
                
        result_dict["pred_probs"].append(torch.cat(utils.gather_tensor(pred_probs)).cpu())
        result_dict["true_labels"].append(torch.argmax(torch.cat(utils.gather_tensor(true_labels)), dim=1).cpu())
    
    pred_probs  = torch.cat([result_dict_batch for result_dict_batch in result_dict["pred_probs"]] , dim=0)
    true_labels = torch.cat([result_dict_batch for result_dict_batch in result_dict["true_labels"]], dim=0)
    
    return pred_probs, true_labels


def log_one_epoch(log_dir_cm, pred_probs, true_labels):
    result_dict = {
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
    }

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


def main():
    # parse args
    args = utils.get_test_args_parser().parse_args()
        
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
    
    # dirs
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.result_dir, args.exp_name)
    result_dir_cm = os.path.join(result_dir, "confusion_matrix")
    result_dir_csv = os.path.join(result_dir, "log_csv")
    result_dir_cam = os.path.join(result_dir, "cam")
    result_dir_cam_correct = os.path.join(result_dir_cam, "correct")
    result_dir_cam_incorrect = os.path.join(result_dir_cam, "incorrect")
        
    if utils.is_main_process():
        for dir_path in [result_dir,
                         result_dir_cm,
                         result_dir_csv,
                         result_dir_cam,
                         result_dir_cam_correct,
                         result_dir_cam_incorrect]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # print args
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
        
    # load data list
    data_list_dict = utils.load_data_list_from_json(args.data_list_json)
    test_list = data_list_dict["test_list"]
    print(f"Data size: test {len(test_list)}")
    
    test_loader = get_dataloader(
        root_dir=args.root_dir,
        data_list=test_list,
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
        pretrained=False
    )
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    utils.load_pretrained_model_weights(model, args.ckpt_path)
    model.eval()
    
    # build cam
    cam = CAM(model=model, target_layers=[model.module.layer4[-1]], use_cuda=True)
    
    # test model
    print("\nStart testing!!")
    pred_probs, true_labels = eval_one_epoch_with_save_cam(cam,
                                                           test_loader, 
                                                           result_dir_cam_correct, 
                                                           result_dir_cam_incorrect, 
                                                           args.classification_threshold)
    # log metrics
    if utils.is_main_process():
        result_dict = log_one_epoch(
            result_dir_cm, 
            pred_probs, 
            true_labels, 
        )
        pd.DataFrame(result_dict, index=[0]).to_csv(os.path.join(result_dir_csv, f"result.csv"), index=False)
        
    print("Done!!\n")


if __name__ == '__main__':
    main()
