import torch
import lightly

from ultralytics import YOLO
from torch import nn

from pathlib import Path


import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from lightly.data import LightlyDataset
from lightly.transforms import MoCoV2Transform, utils

from sslcon.yolo_moco import YOLOMoCo
from config.global_config import config as cfg

class TwoCropsTransform:
    """Create two crops of the same image. Needed for custom augmentations"""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k

def train_moco(yolo, output, imgsz, local_data, moco_ckpt):
    """
    Applying MoCo algorithm to the YOLO framework
    Adapted from: https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_moco_memory_bank.html
    """
    # Default MoCoV2 transforms from the paper
    transform = MoCoV2Transform(input_size=int(cfg.sslcon_config.moco_imgsz))

    if cfg.sslcon_config.moco_custom_augmentation:
        # Applying domain specific transformations: https://docs.pytorch.org/vision/0.9/transforms.html 
        print("Apply custom augmentations")
        custom_transform = T.Compose([
            T.RandomResizedCrop(size=int(cfg.sslcon_config.moco_imgsz), scale=(0.8, 1.0), ratio=(0.9, 1.1)),    # Not cutting that huge parts of the image
            T.RandomHorizontalFlip(p=0.5),                                                                      # Signs can be on left and right
            T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),                              # Slight affine transformations possible
            T.RandomApply(
                [T.RandomChoice([
                    T.RandomAdjustSharpness(sharpness_factor=0.8),
                    T.RandomAdjustSharpness(sharpness_factor=1.4)
                ])
            ], p=0.2),                                                                                          # Auto-Focus or blur the image randomly
            T.RandomApply([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
            ], p=0.5),                                                                                          # Avoiding too high jitter: Not changing signal color!
            T.RandomGrayscale(p=0.05),                                                                          # Different weather and light conditions through color
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),                                                    # Better generalization
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                                  # ImageNet Normalization
        ])

        transform = TwoCropsTransform(custom_transform)

    # Building the dataloader with desired transformations
    dataset_train_moco = LightlyDataset(input_dir=Path(cfg.sslcon_config.moco_dataset_path) if local_data == None else Path(local_data), transform=transform)

    dataloader_train_moco = torch.utils.data.DataLoader(
        dataset_train_moco,
        batch_size=cfg.sslcon_config.moco_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.sslcon_config.moco_num_workers,
    )

    # Pre-training with MoCOv2
    model = YOLOMoCo()
    trainer = pl.Trainer(max_epochs=cfg.sslcon_config.moco_epochs, devices=1, accelerator="gpu")

    if moco_ckpt:
        # Continue pre-training from checkpoint
        trainer.fit(model, dataloader_train_moco, ckpt_path=moco_ckpt)
    else:
        trainer.fit(model, dataloader_train_moco)

    torch.save(model.state_dict(), f"{output}/moco.pt")

    # Loading the pretrained weights into the model
    yolo.model.load(model.backbone)

    path_to_model = f"{output}/pretrained.pt"

    yolo.save(path_to_model)

    return path_to_model