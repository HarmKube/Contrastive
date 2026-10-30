import dataclasses
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from lightlyssl.lightly_config import LightlyConfig
from sslcon.cfg_file import SslConfig as SslConConfig
from detco.train_detco import Config as DetCoConfig
from kfolds.kfolds_config import KfoldsConfig


def _deep_merge_dataclass(dc: Any, update: Dict[str, Any]):
    """Recursively merge dict `update` into dataclass `dc` in‑place.
       If the value for a complex object is a str the method tries to load a YAML from the path in the string and fill the object with it
    """
    for k, v in update.items():
        cur = getattr(dc, k)
        if dataclasses.is_dataclass(cur):
            if type(v) is str:
                load_config_from_yaml(v, cur)
            else:
                _deep_merge_dataclass(cur, v)
        else:
            setattr(dc, k, v)

def load_config_from_yaml(path, config):
    """Filles an object with values from an YAML"""
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user_yaml = yaml.safe_load(f) or {}
        _deep_merge_dataclass(config, user_yaml)

def store_config_to_yaml(path, config):
    """Writes the values from an object to a yaml file"""
    with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dataclasses.asdict(config), f, sort_keys=False)

@dataclass
class Config:
    """Main Configuration class"""
    # SupConfigs
    lightly_config: LightlyConfig
    sslcon_config: SslConConfig
    detco_config: DetCoConfig
    kfolds_config: KfoldsConfig
    # GlobalConfig


    # ---------------------- YAML helpers ----------------------
    def from_yaml(self, path) -> "Config":
        if path:
            with open(path, "r", encoding="utf-8") as f:
                user_yaml = yaml.safe_load(f) or {}
            _deep_merge_dataclass(self, user_yaml)

    def to_yaml(self, path):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, sort_keys=False)

config: Config = Config(
    lightly_config=LightlyConfig(),
    sslcon_config=SslConConfig(),
    detco_config=DetCoConfig(),
    kfolds_config=KfoldsConfig()
)

def config_server(config_folder, log):
    '''
    Funktion to load multiple config files from a folder. 
    Creates an generator object to iterate over the config files.
    On each iteration the config file is loaded and the loaded_config object is updated with the new config.
    '''
    config_files = os.listdir(config_folder)
    log.info(f"Found {len(config_files)} config files in {config_folder}")
    print(f"Found {len(config_files)} config files in {config_folder}")
    for config_file in config_files:
        path = os.path.join(config_folder, config_file)
        config.from_yaml(path)
        yield config_file