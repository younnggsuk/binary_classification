import os
import cv2
import numpy as np
import torch
from torchvision import transforms


class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset for binary classification.
    """
    def __init__(
            self,
            root_dir,
            data_list,
            transform=None,
        ):
        self.root_dir = root_dir
        self.data_list = data_list
        self.transform = transform
        
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
        h, w, _ = cv2.imread(image_path_color, cv2.IMREAD_COLOR).shape
        image = cv2.imread(image_path_depth, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        image = np.expand_dims(image, axis=-1)
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.data_list[idx]["label"])
        label = torch.nn.functional.one_hot(label, num_classes=2)
        
        return image, label, image_path_depth
    
    
def get_dataloader(root_dir, data_list, crop_size, batch_size_per_gpu, num_workers, is_train=True):
    """
    Get data loader.
    """
    if is_train:
        # train data loader
        dataset = CustomDataset(
            root_dir=root_dir,
            data_list=data_list,
            transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
            ]),
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
            transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(), 
            ]),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=torch.utils.data.DistributedSampler(dataset, shuffle=False),
            batch_size=batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=True,
        )
    return dataloader