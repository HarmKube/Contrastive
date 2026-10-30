import torchvision.transforms.v2 as v2
import torch
from typing import Optional, Tuple, Union, List, Dict


class DetCoGlobalTransform:
    """
    Implements the DetCo global transform with full MoCo v2 compatibility and
    additional RandAugment as described in [0].

    Returns two augmented global views of the input image.

    [0] Xu et al., DetCo: https://arxiv.org/abs/2201.10053
    """

    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.4,
        cj_hue: float = 0.1,
        min_scale: float = 0.2,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = 11, 
        sigmas: Tuple[float, float] = (0.1, 1.0),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: Dict[str, List[float]] = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        # DetCo-specific extra:
        randaugment_ops: int = 2,
        randaugment_mag: int = 9,
    ):
        # Core transform list
        transform_list = [
            v2.ToImage(),
            v2.RandomResizedCrop(input_size, scale=(min_scale, 1.0)),
            v2.RandomHorizontalFlip(p=hf_prob),
            v2.RandomVerticalFlip(p=vf_prob),
            v2.RandomApply([
                v2.ColorJitter(
                    brightness=cj_bright * cj_strength,
                    contrast=cj_contrast * cj_strength,
                    saturation=cj_sat * cj_strength,
                    hue=cj_hue * cj_strength
                )
            ], p=cj_prob),
            v2.RandomGrayscale(p=random_gray_scale),
            v2.RandomApply([
                v2.GaussianBlur(kernel_size=kernel_size, sigma=sigmas)
            ], p=gaussian_blur),
            v2.RandAugment(num_ops=randaugment_ops, magnitude=randaugment_mag),
        ]

        if rr_prob > 0.0:
            degrees = rr_degrees if rr_degrees is not None else 90
            transform_list.append(
                v2.RandomApply([v2.RandomRotation(degrees=degrees)], p=rr_prob)
            )

        transform_list.extend([
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize(mean=normalize["mean"], std=normalize["std"]),
        ])

        self.transform = v2.Compose(transform_list)

    def __call__(self, image):
        return self.transform(image)
    


class DetCoLocalTransform:
    """
    DetCo local image transform (no patch extraction).
    Applies PIRL-style pre-transform:
        - Random crop (≥ 60%)
        - Resize to 255×255
        - Flip, jitter, grayscale, blur
        - No patch shuffle or crop – image is returned as [3, 255, 255] tensor.
    """

    def __init__(
        self,
        input_size: int = 255,
        min_scale: float = 0.6,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.4,
        cj_hue: float = 0.1,
        grayscale_prob: float = 0.2,
        blur_prob: float = 0.5,
        kernel_size: Optional[float] = 11,
        sigmas: Tuple[float, float] = (0.1, 1.0),
        hf_prob: float = 0.5,
    ):
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.RandomResizedCrop(input_size, scale=(min_scale, 1.0)),
            v2.Resize((input_size, input_size)),
            v2.RandomHorizontalFlip(p=hf_prob),
            v2.RandomApply([
                v2.ColorJitter(
                    brightness=cj_bright * cj_strength,
                    contrast=cj_contrast * cj_strength,
                    saturation=cj_sat * cj_strength,
                    hue=cj_hue * cj_strength
                )
            ], p=cj_prob),
            v2.RandomGrayscale(p=grayscale_prob),
            v2.RandomApply([
                v2.GaussianBlur(kernel_size=kernel_size, sigma=sigmas)
            ], p=blur_prob)
        ])

    def __call__(self, image):
        return self.transform(image)  # → Tensor [3, 255, 255]
