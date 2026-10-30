from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ColorJitterConfig:
    brightness: float = 0.8
    contrast: float = 0.8
    hue: float = 0.2
    prob: float = 0.8
    saturation: float = 0.4
    strength: float = 0.5
    
    def to_dict(self) -> dict:
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "hue": self.hue,
            "prob": self.prob,
            "saturation": self.saturation,
            "strength": self.strength
        }


@dataclass
class GaussianBlurConfig:
    blur_limit: int = 0
    prob: float = 1.0
    sigmas: List[float] = None
    
    def __post_init__(self):
        if self.sigmas is None:
            self.sigmas = [0.0, 0.1]
    
    def to_dict(self) -> dict:
        return {
            "blur_limit": self.blur_limit,
            "prob": self.prob,
            "sigmas": self.sigmas
        }


@dataclass
class NormalizeConfig:
    mean: List[float] = None
    std: List[float] = None
    
    def __post_init__(self):
        if self.mean is None:
            self.mean = [0.485, 0.456, 0.406]
        if self.std is None:
            self.std = [0.229, 0.224, 0.225]
    
    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std
        }


@dataclass
class RandomFlipConfig:
    horizontal_prob: float = 0.5
    vertical_prob: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "horizontal_prob": self.horizontal_prob,
            "vertical_prob": self.vertical_prob
        }


@dataclass
class RandomResizeConfig:
    max_scale: float = 1.0
    min_scale: float = 0.14
    
    def to_dict(self) -> dict:
        return {
            "max_scale": self.max_scale,
            "min_scale": self.min_scale
        }


@dataclass
class TransformArgsConfig:
    color_jitter: ColorJitterConfig = None
    gaussian_blur: GaussianBlurConfig = None
    image_size: List[int] = None
    normalize: NormalizeConfig = None
    random_flip: RandomFlipConfig = None
    random_gray_scale: float = 0.2
    random_resize: RandomResizeConfig = None
    random_rotation: Optional[object] = None
    solarize: Optional[object] = None
    
    def __post_init__(self):
        if self.color_jitter is None:
            self.color_jitter = ColorJitterConfig()
        if self.gaussian_blur is None:
            self.gaussian_blur = GaussianBlurConfig()
        if self.image_size is None:
            self.image_size = [640, 640]
        if self.normalize is None:
            self.normalize = NormalizeConfig()
        if self.random_flip is None:
            self.random_flip = RandomFlipConfig()
        if self.random_resize is None:
            self.random_resize = RandomResizeConfig()
    
    def to_dict(self) -> dict:
        """Convert TransformArgsConfig to dictionary format"""
        return {
            "color_jitter": self.color_jitter.to_dict(),
            "gaussian_blur": self.gaussian_blur.to_dict(),
            "image_size": self.image_size,
            "normalize": self.normalize.to_dict(),
            "random_flip": self.random_flip.to_dict(),
            "random_gray_scale": self.random_gray_scale,
            "random_resize": self.random_resize.to_dict(),
            "random_rotation": self.random_rotation,
            "solarize": self.solarize
        }


@dataclass
class LightlyConfig():
    epochs = 100
    batch_size = 128
    method = "simclr"
    num_workers = None # Set to None to let lightly determine the number of workers based on available CPU cores
    transform_args: TransformArgsConfig = None
    
    def __post_init__(self):
        if self.transform_args is None:
            self.transform_args = TransformArgsConfig()