"""
Feature-Extractor for object embeddings.

Responsibilities
----------------
* Load images/labels via YoloDetectionDataset
* Run a given backbone in no-grad inference mode
* Use a forward-hook to tap the final feature-map
* ROI-Align every GT box -> fixed-size embedding
* Return (embeddings, labels) as NumPy arrays

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import roi_align

from detco.datasets import YoloDetectionDataset 

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _collate(batch: Sequence[tuple]) -> tuple:
    """
    Custom collate-fn that keeps *variable* length sequences per image.
    Output: imgs[B,3,H,W], list[Tensor[N,4]], list[Tensor[N]], list[(H,W)]
    """
    imgs, boxes, clss, shapes = zip(*batch)
    return torch.stack(imgs, 0), list(boxes), list(clss), list(shapes)


def _normalize_device(dev: str | torch.device | None) -> torch.device:
    """Return a valid torch.device (default: auto-cuda-if-available)."""
    if dev is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev) if isinstance(dev, str) else dev


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@torch.no_grad()
def get_embeddings(
    img_dir: Path,
    lbl_dir: Path,
    *,
    backbone: torch.nn.Module,
    batch_size: int = 4,
    device: str | torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract ROI-aligned embeddings for every GT box in *img_dir/lbl_dir*.

    Parameters
    ----------
    img_dir, lbl_dir : Path
        Folders with images + YOLO-txt label files.
    backbone : torch.nn.Module
        Model already moved to desired device and set to eval().
    batch_size : int
        Inference batch size.
    device : str | torch.device | None
        If None we infer from backbone; otherwise overrides it.

    Returns
    -------
    emb_np : np.ndarray  shape=(N_objects, C)
    lbl_np : np.ndarray  shape=(N_objects,)
    """
    device = _normalize_device(device or next(backbone.parameters()).device)

    dataset = YoloDetectionDataset(img_dir, lbl_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
    )

    # --------------------------------------------------------------------- #
    # Forward-hook to capture last feature map
    # --------------------------------------------------------------------- #
    hook_cache: dict[str, torch.Tensor] = {}

    def _cache_feat(_: torch.nn.Module, __: tuple, output: torch.Tensor) -> None:
        hook_cache["feat"] = output

    hook_handle = backbone[-1].register_forward_hook(_cache_feat)

    # --------------------------------------------------------------------- #
    # Inference loop
    # --------------------------------------------------------------------- #
    embeddings: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []

    for batch_idx, (imgs, boxlist, clslist, shapes) in enumerate(loader):
        hook_cache.clear()                       # erase leftovers
        imgs = imgs.to(device, non_blocking=True)
        _ = backbone(imgs)                       # run forward pass

        feat = hook_cache.pop("feat", None)
        if feat is None:
            hook_handle.remove()
            raise RuntimeError(
                "Forward-hook did not fire – verify backbone[-1] reference."
            )

        _, _, hf, wf = feat.shape
        scale_x = torch.as_tensor([wf / s[1] for s in shapes], device=device)
        scale_y = torch.as_tensor([hf / s[0] for s in shapes], device=device)

        for b_idx, boxes in enumerate(boxlist):
            boxes = boxes.to(device)
            if boxes.numel() == 0:
                continue  # skip images with no GT boxes

            # -------- Convert YOLO-pixel -> feature-map coords ------------- #
            # NOTE: YoloDetectionDataset must already provide ABSOLUTE pixel
            #       coords (x1,y1,x2,y2). Otherwise convert them here!
            scaled = boxes.clone()
            scaled[:, [0, 2]] *= scale_x[b_idx]
            scaled[:, [1, 3]] *= scale_y[b_idx]

            rois = torch.cat(
                (torch.full((scaled.size(0), 1), b_idx, device=device), scaled), 1
            )  # ROIAlign expects (batch_idx,x1,y1,x2,y2)

            pooled = roi_align(
                feat,
                rois,
                output_size=(1, 1),
                spatial_scale=1.0,
                aligned=True,
            )  # → (N,C,1,1)

            embeddings.append(pooled.squeeze(-1).squeeze(-1).cpu())
            labels.append(clslist[b_idx].cpu())

        if batch_idx % 20 == 0:
            logging.debug("Processed %d / %d batches", batch_idx + 1, len(loader))

    hook_handle.remove()  # important: prevent ghost hooks

    if not embeddings:
        raise RuntimeError("No bounding boxes found – embeddings list empty.")

    emb_np = torch.cat(embeddings, 0).numpy()
    lbl_np = torch.cat(labels, 0).numpy()

    logging.info("Collected %d embeddings (dim=%d)", *emb_np.shape)
    return emb_np, lbl_np
