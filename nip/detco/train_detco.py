from __future__ import annotations

import os
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image  # may be used by external modules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO
from lightly.loss import NTXentLoss

from detco.Yolo_Detco import Yolo_Detco  # local user module
from detco.datasets import DatasetCfg, UnlabeledImageDataset

from contextlib import nullcontext
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

# Embedding extraction routine (project-local)
from embeddingxai.extractor import get_embeddings

# Project-local config saver
from config.save_config import save_config


# ---------------------------------------------------------------------------
# 1 · Configuration (single source of truth)
# ---------------------------------------------------------------------------

@dataclass
class ModelCfg:
    """Model configuration related to YOLO and backbone slicing."""
    yolo_cfg: str = "yolov8n.yaml"           # Ultralytics model config or weights
    backbone_layers: int = 10                # Number of layers to keep for the backbone slice
    checkpoint_path: str = "detco_model_checkpoint.pt"  # Base folder for run artifacts


@dataclass
class TrainingCfg:
    """Training hyperparameters and control flags."""
    epochs: int = 50
    patience: int = 30
    lr: float = 2e-4
    weight_decay: float = 0.0
    temperature: float = 0.1
    memory_bank_size: tuple[int, int] = (4096, 128)  # (bank_size, feature_dim)
    loss_global: float = 0.5
    loss_local: float = 0.5
    loss_cross: float = 0.5
    use_amp: bool = True
    enable_profiling: bool = False
    resume_ckpt: str | None = None           # Path to resume run directory (expects /checkpoints)


@dataclass
class EmbeddingEvalCfg:
    """Controls periodic evaluation of backbone embeddings."""
    enabled: bool = False                    # Enable/disable evaluation entirely
    img_dir: str = ""                        # Unlabeled image folder
    lbl_dir: str = ""                        # YOLO label folder (txt files)
    batch_size: int = 8                      # Dataloader batch size for the extractor
    freq: int = 5                            # Evaluate every *n* epochs
    top_k_classes: int = 3                   # Restrict metrics to K most frequent classes (if > 0)
    l2_normalize: bool = True                # L2-normalize embeddings before metrics
    metric: str = "cosine"                   # Metric for silhouette score
    compute_nmi: bool = True                 # Compute NMI in addition to silhouette
    kmeans_init: str = "k-means++"           # KMeans init strategy


@dataclass
class Config:
    """Full experiment configuration."""
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    embedding_eval: EmbeddingEvalCfg = field(default_factory=EmbeddingEvalCfg)
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 2 · Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    folder: Path,
    epoch: int,
    backbone: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    crit_global: nn.ModuleList,
    crit_local: nn.ModuleList,
    crit_cross: nn.ModuleList,
) -> Path:
    """
    Save a training snapshot to <folder>/eXYZ.pt and return the path.

    Persisted state includes:
      - current epoch
      - backbone weights (q_backbone.body)
      - full DetCo wrapper state (incl. q/k branches)
      - optimizer & AMP scaler state
      - NTXent losses (global/local/cross) and their memory banks

    Parameters
    ----------
    folder : Path
        Target directory for checkpoints.
    epoch : int
        Current epoch number (used in filename).
    backbone : nn.Module
        DetCo wrapper; reads q_backbone.body for the pure backbone state.
    optimizer : torch.optim.Optimizer
        Optimizer whose state will be saved.
    scaler : torch.cuda.amp.GradScaler
        AMP scaler state for mixed precision.
    crit_global, crit_local, crit_cross : nn.ModuleList
        Lists of NTXentLoss modules for each stage.

    Returns
    -------
    Path
        Path to the written checkpoint file.
    """
    folder.mkdir(parents=True, exist_ok=True)
    ckpt_path = folder / f"e{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "backbone": backbone.q_backbone.body.state_dict(),
            "model": backbone.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "crit_global": [c.state_dict() for c in crit_global],
            "crit_local": [c.state_dict() for c in crit_local],
            "crit_cross": [c.state_dict() for c in crit_cross],
            "memory_global": [c.memory_bank.bank.clone() for c in crit_global],
            "memory_local": [c.memory_bank.bank.clone() for c in crit_local],
            "memory_cross": [c.memory_bank.bank.clone() for c in crit_cross],
        },
        ckpt_path,
    )
    return ckpt_path


def load_checkpoint(
    path: Path,
    backbone: nn.Module | None,
    model: Yolo_Detco,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    crit_global: nn.ModuleList,
    crit_local: nn.ModuleList,
    crit_cross: nn.ModuleList,
) -> int:
    """
    Load training state from a checkpoint file and return the next epoch.

    Notes
    -----
    If `backbone` is provided, only the backbone weights will be loaded into it
    (useful if you want to export/use just the backbone elsewhere). The full
    model state is loaded into `model` regardless.

    Returns
    -------
    int
        Next epoch index to continue training from (i.e., loaded_epoch + 1).
    """
    state = torch.load(path, map_location="cpu")

    if backbone:
        backbone.load_state_dict(state["backbone"], strict=True)

    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["opt"])
    scaler.load_state_dict(state["scaler"])

    for c, saved in zip(crit_global, state["crit_global"]):
        c.load_state_dict(saved)
    for c, saved in zip(crit_local, state["crit_local"]):
        c.load_state_dict(saved)
    for c, saved in zip(crit_cross, state["crit_cross"]):
        c.load_state_dict(saved)

    for c, mem in zip(crit_global, state["memory_global"]):
        c.memory_bank.bank.copy_(mem)
    for c, mem in zip(crit_local, state["memory_local"]):
        c.memory_bank.bank.copy_(mem)
    for c, mem in zip(crit_cross, state["memory_cross"]):
        c.memory_bank.bank.copy_(mem)

    return state["epoch"] + 1


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """
    Return the checkpoint with the highest epoch index in `ckpt_dir`,
    or `None` if the directory contains no `e*.pt` files.
    """
    ckpts = sorted(ckpt_dir.glob("e*.pt"))
    return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------
# 3 · Patch extraction utility
# ---------------------------------------------------------------------------

def create_patches(
    x: torch.Tensor,
    w: int = 3,
    h: int = 3,
    crop_ratio: tuple[float, float] = (0.8, 1.0),
    antialias: bool = True,
) -> torch.Tensor:
    """
    Split a batch of images into a w×h grid of cells, apply a random crop
    per cell (area ratio between `crop_ratio[0]` and `crop_ratio[1]`),
    then resize back to the cell size. Patches are shuffled per image.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H, W).
    w, h : int
        Grid dimensions (columns, rows). H and W must be divisible by h and w.
    crop_ratio : tuple[float, float]
        Min and max area ratio for the inner random crop within each cell.
    antialias : bool
        Use bilinear antialiasing on upsampling during resize.

    Returns
    -------
    torch.Tensor
        Tensor of shape (B*h*w, C, H/h, W/w) containing the shuffled patches.
    """
    B, C, H, W = x.shape
    if H % h or W % w:
        raise ValueError("H and W must be divisible by the patch grid size.")

    ph, pw = H // h, W // w  # cell height/width

    # 1) Split into grid cells using unfold for efficiency
    cells = (
        x.unfold(2, ph, ph)           # B×C×h×ph×W
         .unfold(3, pw, pw)           # B×C×h×w×ph×pw
         .permute(0, 2, 3, 1, 4, 5)   # B×h×w×C×ph×pw
         .reshape(-1, C, ph, pw)      # (B·h·w) × C × ph × pw
    )

    # 2) Random crop in each cell and resize back to (ph, pw)
    bh, bw = cells.shape[-2:]
    min_r, max_r = crop_ratio
    rand_scale = torch.empty(len(cells), device=x.device).uniform_(min_r, max_r)
    ch = (rand_scale * bh).int()
    cw = (rand_scale * bw).int()
    dy = (torch.rand(len(cells), device=x.device) * (bh - ch)).int()
    dx = (torch.rand(len(cells), device=x.device) * (bw - cw)).int()

    out = torch.empty_like(cells)
    for i, (y0, x0, hh, ww) in enumerate(zip(dy, dx, ch, cw)):
        crop = cells[i, :, y0:y0 + hh, x0:x0 + ww]
        if (hh, ww) != (bh, bw):
            crop = F.interpolate(
                crop.unsqueeze(0),
                size=(bh, bw),
                mode="bilinear",
                align_corners=False,
                antialias=antialias,
            ).squeeze(0)
        out[i] = crop

    # 3) Shuffle patches per image
    out = (
        out.view(B, h * w, C, ph, pw)
           [:, torch.randperm(h * w, device=x.device)]
           .reshape(-1, C, ph, pw)
    )
    return out


# ---------------------------------------------------------------------------
# 4 · Profiler helper
# ---------------------------------------------------------------------------

def make_profiler(cfg: TrainingCfg):
    """Return a real profiler (if enabled) or a no-op context manager."""
    if not cfg.enable_profiling:
        return nullcontext()
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler("./log"),
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    )


# ---------------------------------------------------------------------------
# 5 · Embedding evaluation (silhouette, class filtering, optional NMI)
# ---------------------------------------------------------------------------

def evaluate_embeddings(
    cfg: EmbeddingEvalCfg,
    backbone: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Compute silhouette (and optionally NMI) on backbone embeddings.

    Steps
    -----
    1) Extract embeddings and integer labels using project-local extractor.
    2) Optionally restrict to the top-K most frequent classes.
    3) Optional L2-normalization of embeddings.
    4) Compute silhouette score using `cfg.metric`.
    5) Optionally compute KMeans-based NMI.

    Returns
    -------
    dict[str, float]
        At least {"sil": float}; includes {"nmi": float} if enabled.
    """
    emb, labels = get_embeddings(
        Path(cfg.img_dir),
        Path(cfg.lbl_dir),
        backbone=backbone,
        batch_size=cfg.batch_size,
        device=device,
    )

    if cfg.top_k_classes > 0:
        vals, cnt = np.unique(labels, return_counts=True)
        topk = vals[np.argsort(cnt)[-cfg.top_k_classes:]]
        mask = np.isin(labels, topk)
        emb, labels = emb[mask], labels[mask]

    if cfg.l2_normalize:
        emb = normalize(emb)

    sil = silhouette_score(emb, labels, metric=cfg.metric)
    out: dict[str, float] = {"sil": float(sil)}

    if cfg.compute_nmi:
        k = len(np.unique(labels))
        preds = KMeans(
            n_clusters=k,
            n_init=10,
            init=cfg.kmeans_init,
            random_state=0,
        ).fit_predict(emb)
        nmi = normalized_mutual_info_score(labels, preds)
        out["nmi"] = float(nmi)

    return out


# ---------------------------------------------------------------------------
# 6 · Training helpers
# ---------------------------------------------------------------------------

def _make_memory_bank(cfg: TrainingCfg, device: torch.device) -> NTXentLoss:
    """
    Construct an NTXentLoss with a MoCo-style memory bank.
    """
    return NTXentLoss(cfg.temperature, memory_bank_size=cfg.memory_bank_size).to(device)


def train_one_epoch(
    model: Yolo_Detco,
    dataloader: DataLoader,
    crit_g: nn.ModuleList,
    crit_l: nn.ModuleList,
    crit_c: nn.ModuleList,
    opt: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    cfg: TrainingCfg,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    """
    Train the DetCo wrapper for exactly one epoch and return averaged metrics.

    Returns
    -------
    dict[str, float]
        Keys: {"total", "g", "l", "c"} for total/global/local/cross losses.
    """
    metrics = {"total": 0.0, "g": 0.0, "l": 0.0, "c": 0.0}
    model.train()

    for a1, a2, p1, p2 in tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}"):
        a1 = a1.to(device, non_blocking=True)
        a2 = a2.to(device, non_blocking=True)
        p1 = p1.to(device, non_blocking=True)
        p2 = p2.to(device, non_blocking=True)

        # Flatten the 3×3 patch grid from the dataset pipeline:
        # (B, 9, C, H, W) → (B*9, C, H, W)
        p1 = p1.view(-1, *p1.shape[2:])
        p2 = p2.view(-1, *p2.shape[2:])

        # If you prefer generating patches on-the-fly from the global views,
        # call: p1 = create_patches(a1, 3, 3); p2 = create_patches(a2, 3, 3)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            out = model(a1, a2, p1, p2)
            qg, kg = out["q_global"], out["k_global"]
            ql, kl = out["q_local"], out["k_local"]

            loss_g = torch.zeros((), device=device)
            loss_l = torch.zeros((), device=device)
            loss_c = torch.zeros((), device=device)

            # Aggregate per-stage losses across feature stages
            for i in range(model.n_stages):
                # global ↔ global
                loss_g += crit_g[i](qg[i], kg[i])
                # local  ↔ local
                loss_l += crit_l[i](ql[i], kl[i])
                # local  ↔ global (cross-level)
                loss_c += crit_c[i](ql[i], kg[i])

            loss = (
                cfg.loss_global * loss_g
                + cfg.loss_local * loss_l
                + cfg.loss_cross * loss_c
            )

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        metrics["total"] += float(loss.item())
        metrics["g"] += float(loss_g.item())
        metrics["l"] += float(loss_l.item())
        metrics["c"] += float(loss_c.item())

        # Momentum update for key encoder, if implemented in wrapper
        if hasattr(model, "update_momentum"):
            model.update_momentum()

    nb = len(dataloader)
    return {k: v / nb for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# 7 · Main training routine
# ---------------------------------------------------------------------------

def train_detco(
    cfg: Config,
    return_yolo: bool = False,
    input_yolo: YOLO | None = None,
):
    """
    Train DetCo on an unlabeled dataset using a YOLO backbone.

    Parameters
    ----------
    cfg : Config
        Full configuration object.
    return_yolo : bool
        If True, load best backbone weights into the YOLO model and return it.
    input_yolo : YOLO | None
        Optional pre-initialized YOLO model. If None, a new one is created
        based on `cfg.model.yolo_cfg`.

    Returns
    -------
    dict | None
        If `return_yolo` is True, returns a dict with:
          - "yolo": YOLO model with loaded backbone weights
          - "best_ckpt": path to best checkpoint
          - "final_model": path to saved YOLO weights
          - "run_dir": base run directory
        Otherwise returns None.

    Example
    -------
    >>> cfg = Config()
    >>> result = train_detco(cfg, return_yolo=True)
    >>> if result:
    ...     print(result["final_model"])
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Create or resume a run directory
    if cfg.training.resume_ckpt:
        print("Resuming from:", cfg.training.resume_ckpt)
        run_dir = Path(cfg.training.resume_ckpt)
        if not run_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory {run_dir} does not exist.")
    else:
        run_dir = Path(cfg.model.checkpoint_path) / f"train_detco_{datetime.now():%Y%m%d-%H%M%S}"
        save_config(cfg, run_dir / "config.yaml")

    ckpt_dir = run_dir / "checkpoints"

    if cfg.training.resume_ckpt:
        best_ckpt = latest_checkpoint(ckpt_dir)
    else:
        best_ckpt: Path | None = None

    # Build YOLO and slice its backbone
    yolo = input_yolo or YOLO(cfg.model.yolo_cfg)
    backbone = yolo.model.model[: cfg.model.backbone_layers]
    model = Yolo_Detco(backbone).to(device)

    # Optimizer, AMP scaler, and contrastive losses per stage
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.use_amp)
    crit_g = nn.ModuleList([_make_memory_bank(cfg.training, device) for _ in range(model.n_stages)])
    crit_l = nn.ModuleList([_make_memory_bank(cfg.training, device) for _ in range(model.n_stages)])
    crit_c = nn.ModuleList([_make_memory_bank(cfg.training, device) for _ in range(model.n_stages)])

    # Resume training if requested
    if cfg.training.resume_ckpt and best_ckpt is not None:
        start_epoch = load_checkpoint(best_ckpt, None, model, opt, scaler, crit_g, crit_l, crit_c)
    else:
        start_epoch = 1

    # Data
    ds = UnlabeledImageDataset(cfg.dataset)
    dl_kwargs: Dict[str, Any] = {
        "batch_size": cfg.dataset.batch_size,
        "shuffle": cfg.dataset.shuffle,
        "pin_memory": True,
    }
    if cfg.dataset.num_workers > 0:
        dl_kwargs.update(
            {
                "num_workers": cfg.dataset.num_workers,
                "persistent_workers": cfg.dataset.persistent_workers,
            }
        )
    dl = DataLoader(ds, **dl_kwargs)

    best_loss, patience = float("inf"), 0

    # Optional profiling context
    with make_profiler(cfg.training) as prof:
        for epoch in range(start_epoch, cfg.training.epochs + 1):
            # 1) Train one epoch
            metrics = train_one_epoch(
                model, dl, crit_g, crit_l, crit_c, opt, scaler, cfg.training, device, epoch, cfg.training.epochs
            )
            print(
                f"Epoch {epoch:03d} | total={metrics['total']:.4f} | g={metrics['g']:.4f} "
                f"| l={metrics['l']:.4f} | c={metrics['c']:.4f}"
            )

            # 2) Periodic embedding evaluation (backbone only)
            if cfg.embedding_eval.enabled and epoch % cfg.embedding_eval.freq == 0:
                backbone_eval = model.q_backbone.body  # excludes MLP heads
                metrics_eval = evaluate_embeddings(cfg.embedding_eval, backbone_eval, device)
                msg = (
                    f"Embeddings @ epoch {epoch} — Silhouette ({cfg.embedding_eval.metric}): "
                    f"{metrics_eval['sil']:.4f}"
                )
                if "nmi" in metrics_eval:
                    msg += f" | NMI: {metrics_eval['nmi']:.4f}"
                print(msg)

            # 3) Checkpointing & early stopping
            if metrics["total"] < best_loss:
                best_loss, patience = metrics["total"], 0
                best_ckpt = save_checkpoint(
                    folder=ckpt_dir,
                    epoch=epoch,
                    backbone=model,
                    optimizer=opt,
                    scaler=scaler,
                    crit_global=crit_g,
                    crit_local=crit_l,
                    crit_cross=crit_c,
                )
                print(f"✓ checkpoint saved → {best_ckpt}")
            else:
                patience += 1
                if patience >= cfg.training.patience:
                    print("Early stopping.")
                    break

    # Optional: print profiler summaries
    if cfg.training.enable_profiling:
        print("\nTop CPU ops (total time):")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
        print("\nTop CPU ops (self time):")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
        print("\nTop CUDA ops:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # Optional: return YOLO with the best backbone weights loaded
    if return_yolo:
        if best_ckpt is None:
            raise RuntimeError("No checkpoint has been saved; cannot return YOLO model.")

        checkpoint = torch.load(best_ckpt, map_location="cpu")
        backbone = yolo.model.model[: cfg.model.backbone_layers]
        backbone.load_state_dict(checkpoint["backbone"], strict=True)

        final_model_path = run_dir / "final_yolo_model.pt"
        yolo.save(final_model_path)

        return {
            "yolo": yolo,
            "best_ckpt": str(best_ckpt),
            "final_model": str(final_model_path),
            "run_dir": str(run_dir),
        }
