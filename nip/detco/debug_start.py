import argparse
import dataclasses

import yaml
from detco.train_detco import Config, train_detco
from config.global_config import load_config_from_yaml


def cfg_from_cli() -> Config:
    """Return a `Config` that merges defaults, optional YAML and CLI flags."""
    parser = argparse.ArgumentParser(description="Train DETCO with a YOLOv8 backbone.")
    parser.add_argument("--config", "-c", type=str, help="YAML file with overrides")

    # generate flat flags like "--training.epochs"
    def _add_flags(prefix: str, subdc):
        for key, val in subdc.__dict__.items():
            flag = f"--{prefix}{key}"  # e.g. --training.epochs
            parser.add_argument(flag, type=type(val), default=None)
    base = Config()
    for group, subdc in base.__dict__.items():
        if dataclasses.is_dataclass(subdc):
            _add_flags(f"{group}.", subdc)
        else:  # device, seed (plain attrs)
            parser.add_argument(f"--{group}", type=type(subdc), default=None)

    args = parser.parse_args()    
    cfg = Config()
    load_config_from_yaml(args.config, cfg)

    # apply CLI overrides
    for k, v in vars(args).items():
        if k == "config" or v is None:
            continue
        if "." in k:
            group, key = k.split(".", 1)
            setattr(getattr(cfg, group), key, v)
        else:
            setattr(cfg, k, v)

    return cfg

"""File to quick start the Detco training with easy config overrides"""
if __name__ == "__main__":
    cfg = cfg_from_cli()
    print("Using configuration:\n" + yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
    train_detco(cfg, return_yolo=True)