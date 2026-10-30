from dataclasses import dataclass


@dataclass
class KfoldsConfig():
    """
    Setting the configuration for the YOLO-kfolds model.
    """
    kfolds_num_workers = 4
    kfolds_batch_size = 256
    kfolds_memory_bank_size = 4096
    kfolds_epochs = 200
    kfolds_imgsz = 640
    kfolds_lr = 0.03
    kfolds_momentum = 0.9
    kfolds_weight_decay = 0.0001
    
    seed = 1
    
    yolo_gerald_epochs = 200
    yolo_coco_epochs = 100
    yolo_version = "yolov8n.yaml"
    yolo_backbone_end = 10 if "yolov8n" in yolo_version else 11
    yolo_freeze_backbone = False
    yolo_batch_size = 16
