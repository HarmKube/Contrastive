import torch
import lightly

from ultralytics import YOLO
from torch import nn
from pathlib import Path

import copy

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models import ResNetGenerator
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)
from lightly.transforms import MoCoV2Transform, utils

from config.global_config import config as cfg

class YOLOMoCo(pl.LightningModule):
    """
    Applying Projection Head after YOLO backbone
    Adapted from: https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_moco_memory_bank.html
    """
    def __init__(self):
        super(YOLOMoCo, self).__init__()
        if ".yaml" in cfg.sslcon_config.yolo_version:
            yolo = YOLO(cfg.sslcon_config.yolo_version)
        else:
            yolo = YOLO(Path(cfg.sslcon_config.yolo_version))

        # Inserting the backbone of YOLO
        yolo.model.model = yolo.model.model[:cfg.sslcon_config.yolo_backbone_end]
        self.backbone = yolo.model.requires_grad_()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Create the moco model with (momentum) encoder
        self.projection_head = MoCoProjectionHead(input_dim=256, hidden_dim=512, output_dim=128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.pool_momentum = copy.deepcopy(self.pool)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        # Setting query-encoder to train mode
        self.backbone.train()
        self.backbone.requires_grad_()
        self.projection_head.train()
        self.projection_head.requires_grad_()

        # No gradients for key-encoder
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.pool_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # Using the contrastive loss
        self.criterion = NTXentLoss(
            temperature=0.1, memory_bank_size=(cfg.sslcon_config.moco_memory_bank_size, 128)
        )

    def training_step(self, batch, batch_idx):
        (x_q, x_k), _, _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        # Get the queries with query-encoder
        q = self.backbone(x_q)
        q = self.pool(q).flatten(start_dim=1)
        q = self.projection_head(q)

        # Get the keys with key-encoder
        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k)
        k = self.pool_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        # Calculating contrastive loss
        loss = self.criterion(q, k)
        self.log("train_loss_ssl", loss)
        return loss

    def on_train_epoch_end(self):
        self.custom_histogram_weights()

    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def configure_optimizers(self):
        """
        Using an SGD optimizer with the MultiStepLR from the MoCo paper.
        """
        optim = torch.optim.SGD(
            self.parameters(),
            lr=cfg.sslcon_config.moco_lr,
            momentum=cfg.sslcon_config.moco_momentum,
            weight_decay=cfg.sslcon_config.moco_weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[120,160], gamma=0.1)
        return [optim], [scheduler]
    