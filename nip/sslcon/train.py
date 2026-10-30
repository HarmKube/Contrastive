import torch
import lightly
import yaml

from ultralytics import YOLO
from pathlib import Path

from lightlyssl.train import train_lightly
from sslcon.train_moco import train_moco
from config.global_config import config as cfg
from sslcon.tools import compute_stats
from detco.train_detco import train_detco
from yt_gerald.train_yt_gerald import train_yt

from benchmarks.val import analyze_frequencies


# Loading the base model for our training
def train(data="/mysrc/nip/sslcon/geraldsign_local.yaml", imgsz=1920, output="/mysrc/nip/sslcon/runs", ft_coco=False, ssl_mode=None, local_data=None, moco_ckpt=None, yolo_ckpt=None, lightly_ckpt=False):
    """
    Root for all pre-training procedures. First the YOLO model is pre-trained with one of the available pre-training
    methods. Next, a freezing step can be applied before fine-tuning on the GERALD-dataset.
    Args:
      data PATH: The configuration-file for GERALD-training
      imgsz INT: Image size for training
      output PATH: Path the YOLO evaluation is written to.
      ft_coco: Specify if you want to pre-train on COCO manually (dataset needs to be downloaded).
      ssl_mode MODE: e.g., `YOLOMoCo`, `yolo_detco`, `lightly`, `generic`)
      local_data PATH: Internal flag for testing.
      moco_ckpt PATH: Continue training on previous lightning checkpoint for MoCo.
      yolo_ckpt PATH: Continue from YOLO checkpoint.
      lightly_ckpt: Continue training on previous lightly checkpoint for other methods.
    """
    data = Path(data)
    output = Path(output)

    # Creating the base model
    naming = "PT" if ".pt" in cfg.sslcon_config.yolo_version else "YAML"
    yolo = None
    if ".yaml" in cfg.sslcon_config.yolo_version:
        yolo = YOLO(cfg.sslcon_config.yolo_version)
    else:
        yolo = YOLO(f"{output}/{cfg.sslcon_config.yolo_version}")

    current_yolo_path = ""

    # Selecting the Pre-Training. Model weights are saved in current_yolo_path.
    if ssl_mode == "YOLOMoCo":
        print(f"\nStart pretraining with {ssl_mode}.")
        if yolo_ckpt == None:
            current_yolo_path = train_moco(yolo, output=output, imgsz=imgsz, local_data=local_data, moco_ckpt=moco_ckpt)
        naming += f"_YOLOMoCo_Aug{'Custom' if cfg.sslcon_config.moco_custom_augmentation else 'Df'}"
    elif ssl_mode == "yolo_detco":
        print("Start training Yolo Backbone with Detco")
        if not cfg.detco_config:
            raise ValueError("Please provide a valid DetCo configuration.")

        result = train_detco(cfg.detco_config, input_yolo=yolo, return_yolo=True)
        current_yolo_path = result["final_model"]
        naming += "_YOLODetCo"
    elif ssl_mode == "lightly":
        if not cfg.lightly_config:
            raise ValueError("Please provide a valid lightly configuration.")
        current_yolo_path = train_lightly(model=yolo, output=output, data=local_data, resume=lightly_ckpt)
        naming += "_lightly"
    elif ssl_mode == "Generic":
        naming += "_Generic"
        if yolo_ckpt == None:
            yolo.train(data="/mysrc/data/generic_gerald.yaml", epochs=cfg.sslcon_config.yolo_gerald_epochs, imgsz=imgsz, project=Path(f"{output}/{naming}"), name="GERALD_Generic", batch=cfg.sslcon_config.yolo_batch_size)
        current_yolo_path = f"{output}/{naming}/GERALD_Generic/weights/best.pt"
    elif ssl_mode == None:
        if ".yaml" in cfg.sslcon_config.yolo_version:
            current_yolo_path = cfg.sslcon_config.yolo_version
        else:
            current_yolo_path = f"{output}/{cfg.sslcon_config.yolo_version}"
        print(f"No SSL mode selected, using {cfg.sslcon_config.yolo_version} as is. No pre-training applied.")
    else:
        raise ValueError(f"Unknown SSL mode: {ssl_mode}. Please choose from one or None for no pre-training.")

    resume = False
    # Loading potential checkpoints or otherwise pre-trained models
    if not yolo_ckpt == None:
        print("\nLoading YOLO checkpoint.")
        yolo = YOLO(yolo_ckpt)
        if "last.pt" in yolo_ckpt:
            resume = True
    elif ssl_mode:
        print("\nUsing pretrained weights.")
        yolo = YOLO(current_yolo_path)
    else:
        print("\nUsing no pretraining.")

    freeze = None
    # Optional: Freezing the backbone
    if cfg.sslcon_config.yolo_freeze_backbone and not resume:
        #yolo.add_callback("on_train_start", freeze_layer)
        freeze = 10
        naming += "_Freeze"
        results = yolo.train(data=data, epochs=cfg.sslcon_config.yolo_gerald_epochs, imgsz=imgsz, project=Path(f"{output}/{naming}"), name="GERALD_FREEZE", batch=cfg.sslcon_config.yolo_batch_size, freeze=freeze)
        yolo = YOLO(Path(f"{output}/{naming}/GERALD_FREEZE/weights/best.pt"))
    elif cfg.sslcon_config.yolo_freeze_backbone and resume:
        naming += "_Freeze"

    # Optional: If not pre-trained weights from ultralytics -> Now pre-train on COCO
    if ft_coco:
        yolo.train(model=current_yolo_path, data="/mysrc/nip/sslcon/custom_coco.yaml", epochs=cfg.sslcon_config.yolo_coco_epochs, imgsz=640, batch=cfg.sslcon_config.yolo_batch_size, project=output, name=f"{naming}_COCO")
        naming += "_COCO"
        current_weight_path += "weights/best.pt"
        yolo = YOLO(Path(f"{output}/{naming}/weights/best.pt"))

    # Fine-Tune on GERALD data
    if ft_coco:
        results = yolo.train(data=data, epochs=cfg.sslcon_config.yolo_gerald_epochs, imgsz=imgsz, project=Path(f"{output}/{naming}"), name="GERALD", batch=cfg.sslcon_config.yolo_batch_size)
        naming += "/GERALD"
        val_results = yolo.val(model=Path(f"{output}/{naming}/weights/best.pt"), data=data, imgsz=imgsz, batch=cfg.sslcon_config.yolo_batch_size, split="test", project=Path(f"{output}/{naming}"), name=f"VAL")
    else:
        results = yolo.train(data=data, epochs=cfg.sslcon_config.yolo_gerald_epochs, imgsz=imgsz, project=Path(f"{output}/{naming}"), name="GERALD", batch=cfg.sslcon_config.yolo_batch_size, resume=resume)
        naming += "/GERALD"
        yolo = YOLO(Path(f"{output}/{naming}/weights/best.pt"))
        val_results = yolo.val(data=data, imgsz=imgsz, batch=cfg.sslcon_config.yolo_batch_size, split="test", project=Path(f"{output}/{naming}"), name=f"VAL")

    # Compute stats and analyze per-class results
    with open(data, 'r') as f:
        data_config = yaml.safe_load(f)
    data_path = data_config['path']

    #data_path = f"{SETTINGS['datasets_dir']}/{data_path}"

    compute_stats(val_results, data_path, f"{output}/{naming}/VAL")
    analyze_frequencies(f"{output}/{naming}/VAL")
