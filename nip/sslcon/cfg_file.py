from dataclasses import dataclass


@dataclass
class SslConfig():
    """
    Setting the configuration for the YOLO-MoCo model.
    """
    moco_dataset_path = "/myconfigs/pretrain_yt/images"

    moco_num_workers = 8
    moco_batch_size = 256
    moco_memory_bank_size = 4096
    moco_epochs = 200
    moco_imgsz = 640
    moco_lr = 0.03
    moco_momentum = 0.9
    moco_weight_decay = 0.0001
    moco_custom_augmentation = False
    
    seed = 1
    
    yolo_gerald_epochs = 200
    yolo_coco_epochs = 100
    yolo_version = "yolov8n.yaml"
    yolo_backbone_end = 10 if "yolov8n" in yolo_version else 11
    yolo_freeze_backbone = False
    yolo_batch_size = 16
    yolo_yt_epochs = 200
