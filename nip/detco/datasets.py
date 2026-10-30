from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset
from torchvision.io import read_image

import numpy as np
from typing import Callable, Optional
import random

from detco.detco_transforms import DetCoGlobalTransform, DetCoLocalTransform


@dataclass
class AugmentCfg:
    input_size_global: int = 224
    input_size_local: int = 255
    cj_prob: float = 0.8
    cj_strength: float = 1.0
    cj_bright: float = 0.4
    cj_contrast: float = 0.4
    cj_sat: float = 0.4
    cj_hue: float = 0.1
    min_scale_global: float = 0.2
    min_scale_local: float = 0.6
    random_gray_scale: float = 0.2
    gaussian_blur: float = 0.5
    kernel_size: Optional[int] = 11
    sigmas: Tuple[float, float] = (0.1, 1.0)
    vf_prob: float = 0.0
    hf_prob: float = 0.5
    rr_prob: float = 0.0
    rr_degrees: Optional[Union[float, Tuple[float, float]]] = None
    randaugment_ops: int = 2
    randaugment_mag: int = 9
    local_patch_size = 65
    grid_cell_width = 85

@dataclass
class DatasetCfg:
    root: Path = Path("./data/images/unlabeled")
    batch_size: int = 32
    num_workers: int = 8
    shuffle: bool = True
    persistent_workers: bool = True

    augment: AugmentCfg = field(default_factory=AugmentCfg)

def build_detco_transforms(cfg: AugmentCfg):
    global_transform = DetCoGlobalTransform(
        input_size=cfg.input_size_global,
        cj_prob=cfg.cj_prob,
        cj_strength=cfg.cj_strength,
        cj_bright=cfg.cj_bright,
        cj_contrast=cfg.cj_contrast,
        cj_sat=cfg.cj_sat,
        cj_hue=cfg.cj_hue,
        min_scale=cfg.min_scale_global,
        random_gray_scale=cfg.random_gray_scale,
        gaussian_blur=cfg.gaussian_blur,
        kernel_size=cfg.kernel_size,
        sigmas=cfg.sigmas,
        vf_prob=cfg.vf_prob,
        hf_prob=cfg.hf_prob,
        rr_prob=cfg.rr_prob,
        rr_degrees=cfg.rr_degrees,
        randaugment_ops=cfg.randaugment_ops,
        randaugment_mag=cfg.randaugment_mag,
    )

    local_transform = DetCoLocalTransform(
        input_size=cfg.input_size_local,
        min_scale=cfg.min_scale_local,
        cj_prob=cfg.cj_prob,
        cj_strength=cfg.cj_strength,
        cj_bright=cfg.cj_bright,
        cj_contrast=cfg.cj_contrast,
        cj_sat=cfg.cj_sat,
        cj_hue=cfg.cj_hue,
        grayscale_prob=cfg.random_gray_scale,
        blur_prob=cfg.gaussian_blur,
        kernel_size=cfg.kernel_size,
        sigmas=cfg.sigmas,
        hf_prob=cfg.hf_prob,
    )

    return global_transform, local_transform


class UnlabeledImageDataset(Dataset):
    def __init__(self, cfg):
        # --------------------------------------------------------------- #
        # 0) File loading
        # --------------------------------------------------------------- #
        self.root = Path(cfg.root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root {self.root} not found")

        self.files = [p for p in self.root.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}]

        self.a_config = cfg.augment  # Alias für Augmentierungseinstellungen

        self.t_global, self.t_local_base = build_detco_transforms(self.a_config)

        self.normalize = v2.Compose([
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def _create_local_patches(self, img: torch.Tensor, grid_cell_width = 85, patch_size = 65) -> list[torch.Tensor]:
        """
        Implements PIRL-style grid permutation + crop for DetCo local view.
        Returns 9 shuffled 64x64 patches from a resized 255x255 crop.
        """

        # Apply local base transforms (blur, jitter etc.)
        img_aug = self.t_local_base(img)

        # Divide into 3×3 grids of 85×85
        patches = []
        for i in range(3):
            for j in range(3):
                patch = img_aug[:, i*grid_cell_width:(i+1)*grid_cell_width, j*grid_cell_width:(j+1)*grid_cell_width]
                # Random crop of each patch to 64×64 to break continuity
                top = random.randint(0, grid_cell_width - patch_size)
                left = random.randint(0, grid_cell_width - patch_size)
                patch_crop = patch[:, top:top+patch_size, left:left+patch_size]
                patches.append(patch_crop)

        # Shuffle patches
        random.shuffle(patches)
        return torch.stack([self.normalize(p) for p in patches])

    def __getitem__(self, idx: int):
        img = read_image(str(self.files[idx]))[:3].float() / 255.0  # → float32 in [0,1]

        # 2 Global Views
        global1 = self.t_global(img)
        global2 = self.t_global(img)

        # 2 Local Views (9 Patches jeweils)
        local1 = self._create_local_patches(img, 
                                            grid_cell_width=self.a_config.grid_cell_width,
                                            patch_size=self.a_config.local_patch_size)
        local2 = self._create_local_patches(img,
                                            grid_cell_width=self.a_config.grid_cell_width,
                                            patch_size=self.a_config.local_patch_size)

        return global1, global2, local1, local2


# -----------------------------------------------------------------------------
# Dataset & dataloader
# -----------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class YoloDetectionDataset(Dataset):
    """
    Read images and YOLO-format label files (x_center, y_center, w, h).
    
    Parameters
    ----------
    img_dir : Path
        Directory containing the images.
    lbl_dir : Path
        Directory containing the matching *.txt label files.
    transform : Callable[[torch.Tensor], torch.Tensor] | None, default None
        Additional transform(s) to apply after basic float conversion and
        (optional) normalization. Example: v2.Compose([...]).
    normalize : bool, default True
        If True, apply ImageNet mean/std normalization – must match training.
    """

    def __init__(
        self,
        img_dir: Path,
        lbl_dir: Path,
        *,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize: bool = True,
    ):
        self.img_paths: List[Path] = sorted(
            [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
        self.lbl_dir = lbl_dir
        self.transform = transform

        # build Normalize transform once
        self.normalize = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD) if normalize else None

    # ------------------------------------------------------------
    # PyTorch dataset interface
    # ------------------------------------------------------------
    def __len__(self) -> int:                     # type: ignore[override]
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor, Tuple[int, int]]:  # type: ignore[override]
        img_path = self.img_paths[idx]

        # ---- image -----------------------------------------------------------------
        img = read_image(str(img_path)).float() / 255.0              # (C,H,W) in [0,1]
        if self.normalize:
            img = self.normalize(img)                               # match training stats
        if self.transform:
            img = self.transform(img)                               # any extra aug / resize

        H, W = img.shape[-2:]                                        # after transforms!

        # ---- labels ----------------------------------------------------------------
        txt_file = self.lbl_dir / f"{img_path.stem}.txt"
        if txt_file.is_file():
            lbl_arr = np.loadtxt(txt_file, ndmin=2)                  # shape (N,5)
        else:
            lbl_arr = np.empty((0, 5), dtype=float)

        if lbl_arr.size:
            cls = torch.as_tensor(lbl_arr[:, 0], dtype=torch.long)   # (N,)
            rel = torch.as_tensor(lbl_arr[:, 1:], dtype=torch.float32)  # 0-1
            boxes_whxy = rel.clone()
            boxes_whxy[:, [0, 2]] *= W
            boxes_whxy[:, [1, 3]] *= H

            xyxy = torch.zeros_like(boxes_whxy)                      # (N,4)
            xyxy[:, 0] = boxes_whxy[:, 0] - boxes_whxy[:, 2] / 2      # x1
            xyxy[:, 1] = boxes_whxy[:, 1] - boxes_whxy[:, 3] / 2      # y1
            xyxy[:, 2] = boxes_whxy[:, 0] + boxes_whxy[:, 2] / 2      # x2
            xyxy[:, 3] = boxes_whxy[:, 1] + boxes_whxy[:, 3] / 2      # y2
        else:
            cls = torch.zeros((0,), dtype=torch.long)
            xyxy = torch.zeros((0, 4), dtype=torch.float32)

        # return everything the rest of the pipeline needs
        return img, xyxy, cls, (H, W)