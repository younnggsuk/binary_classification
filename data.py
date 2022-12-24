import os
import cv2
import torch
import random
import numpy as np
from typing import Tuple
from numpy import ndarray

from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F


class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(
            self,
            root_dir,
            data_list,
            crop_size,
            is_train
        ):
        self.root_dir = root_dir
        self.data_list = data_list
        self.crop_size = crop_size
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        image_path_color = os.path.join(
            self.root_dir,
            self.data_list[idx]["color_path"]
        )
        image_path_depth = os.path.join(
            self.root_dir,
            self.data_list[idx]["depth_path"]
        )
        
        image_color = cv2.imread(image_path_color, cv2.IMREAD_COLOR)
        image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
        h, w, _ = image_color.shape
        
        image_depth = cv2.imread(image_path_depth, cv2.IMREAD_GRAYSCALE)
        image_depth = cv2.resize(image_depth, (w, h), interpolation=cv2.INTER_LINEAR)
        image_depth = np.expand_dims(image_depth, axis=-1)
        
        if self.is_train:
            crop_pos = get_crop_params((h, w), (self.crop_size, self.crop_size))
            apply_flip = random.random() > 0.5
            color_jitter_params = get_color_jitter_params()

            transform_color = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Lambda(lambda image: _crop(image, crop_pos, (self.crop_size, self.crop_size))),
                transforms.Lambda(lambda image: _flip(image, apply_flip)),
                transforms.Lambda(lambda image: _color_jitter(image, color_jitter_params)),
                transforms.ToTensor(),
            ])
            transform_depth = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Lambda(lambda image: _crop(image, crop_pos, (self.crop_size, self.crop_size))),
                transforms.Lambda(lambda image: _flip(image, apply_flip)),
                transforms.Lambda(lambda image: _color_jitter(image, color_jitter_params)),
                transforms.ToTensor(),
            ])
        else:
            transform_color = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
            ])
            transform_depth = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
            ])
        
        image_color = transform_color(image_color)
        image_depth = transform_depth(image_depth)
        image_rgbd = torch.cat([image_color, image_depth], dim=0)
        
        label = torch.tensor(self.data_list[idx]["label"])
        label = torch.nn.functional.one_hot(label, num_classes=2)
        
        return image_rgbd, label, image_path_color


def get_dataloader(root_dir, data_list, crop_size, batch_size_per_gpu, num_workers, is_train=True):
    """
    Get data loader.
    """
    if is_train:
        # train data loader
        dataset = CustomDataset(
            root_dir=root_dir,
            data_list=data_list,
            crop_size=crop_size,
            is_train=is_train,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=torch.utils.data.DistributedSampler(dataset, shuffle=True),
            batch_size=batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        # validation data loader
        dataset = CustomDataset(
            root_dir=root_dir,
            data_list=data_list,
            crop_size=crop_size,
            is_train=is_train,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=torch.utils.data.DistributedSampler(dataset, shuffle=False),
            batch_size=batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=True,
        )
    return dataloader


def _flip(
        image: Image,
        apply_flip: bool
    ) -> Image:
    
    if apply_flip:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def get_crop_params(
        image_size: Tuple[int, int], # (h, w)
        crop_size: Tuple[int, int]   # (h, w)
    ) -> Tuple[int, int]: 
    
    h, w = image_size
    new_h, new_w = crop_size
    
    x_pos = random.randint(0, np.maximum(0, w - new_w))
    y_pos = random.randint(0, np.maximum(0, h - new_h))
    return (x_pos, y_pos)


def _crop(
        image: Image,
        crop_pos: Tuple[int, int],
        crop_size: Tuple[int, int]
    ) -> Image:
    
    w, h = image.size
    x, y = crop_pos
    new_w, new_h = crop_size
    
    if (w > new_w or h > new_h):
        return image.crop((x, y, x + new_w, y + new_h))
    return image


def get_color_jitter_params(
        brightness=0.4, 
        contrast=0.4, 
        saturation=0.2, 
        hue=0.1
    ):
    
    color_jitter_params = transforms.ColorJitter.get_params(
        brightness = [max(0, 1 - brightness), 1 + brightness],
        contrast   = [max(0, 1 - contrast),   1 + contrast  ], 
        saturation = [max(0, 1 - saturation), 1 + saturation],
        hue        = [-hue, hue]
    )
    return color_jitter_params


def _color_jitter(
        image: Image,
        params: Tuple[int, float, float, float, float]
    ) -> Image:
    
    fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = params
    for fn_id in fn_idx:
        if fn_id == 0 and brightness_factor is not None:
            image = F.adjust_brightness(image, brightness_factor)
        elif fn_id == 1 and contrast_factor is not None:
            image = F.adjust_contrast(image, contrast_factor)
        elif fn_id == 2 and saturation_factor is not None:
            image = F.adjust_saturation(image, saturation_factor)
        elif fn_id == 3 and hue_factor is not None:
            image = F.adjust_hue(image, hue_factor)
    return image