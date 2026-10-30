from ultralytics import YOLO
import os
from config.global_config import config as cfg

def train_yt(yolo, data="./data/pretrain_yt.yaml", imgsz=1080, output="/mysrc/nip/sslcon/runs"):
    """
    Pretraining YOLO with labeled YouTube dataset
    """
    print(output)
    
    model_path = f'{output}/pretrain_yt/weights/last.pt'
    # resume training in case last.pt exists
    if os.path.exists(model_path):
        yolo = YOLO(model_path)
        yolo.train(data=data, imgsz=imgsz, epochs=200, batch=cfg.sslcon_config.yolo_batch_size, name='pretrain_yt', exist_ok=True, resume = True, project=output)
    else:
        yolo.train(data=data, imgsz=imgsz, epochs=200, batch=cfg.sslcon_config.yolo_batch_size, name='pretrain_yt', exist_ok=False, project=output)

    path_to_model = f"{output}/pretrain_yt/weights/best.pt"
    
    return path_to_model