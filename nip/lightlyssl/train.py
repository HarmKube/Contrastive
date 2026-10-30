import lightly_train
from ultralytics import YOLO
import yaml
from pathlib import Path
from config.global_config import config as cfg

def train_lightly(model:YOLO, data:str, output:str, resume:bool):
    # Handle data parameter - only support YAML files
    data_path = Path(data)
    if not data_path.suffix.lower() in ['.yaml', '.yml']:
        raise ValueError(f"Data parameter must be a YAML file, got {data_path}")
    
    # Load YAML configuration
    print(f"Loading dataset configuration from {data_path}")
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get base directory
    base_dir = Path(data_config.get('path', '.'))
    
    # Find the images directory
    images_dir = base_dir / "images"
    if not images_dir.exists():
        raise ValueError(f"Images directory not found at {images_dir}")
    
    lightly_train.train(out=output, 
    data=images_dir, 
    model=model, 
    method=cfg.lightly_config.method, 
    epochs=cfg.lightly_config.epochs, 
    batch_size=cfg.lightly_config.batch_size, 
    resume_interrupted=resume, 
    num_workers=cfg.lightly_config.num_workers, 
    transform_args = cfg.lightly_config.transform_args.to_dict())

    return f"{output}/my_experiment/exported_models/exported_last.pt"