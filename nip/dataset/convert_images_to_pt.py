#!/usr/bin/env python3
"""
convert_images_to_pt.py
========================
Konvertiert Bilddateien in Tensorformate (.pt) für schnelleren Trainingseinstieg.
Beispiel:
    python convert_images_to_pt.py --input-dir ./data/images --output-dir ./data/pt
"""

import argparse
from pathlib import Path
import torch
from torchvision.transforms import functional as TF
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def convert_images(input_dir: Path, output_dir: Path, overwrite: bool = False) -> None:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_exts]

    for img_path in tqdm(files, desc="Converting images"):
        out_path = output_dir / (img_path.stem + ".pt")
        if out_path.exists() and not overwrite:
            continue

        try:
            img = Image.open(img_path).convert("RGB")  # robust gegen defekte JPEGs
            tensor = TF.to_tensor(img)  # → FloatTensor, [0, 1], shape [C, H, W]
            torch.save(tensor, out_path)
        except (UnidentifiedImageError, OSError) as e:
            print(f"[!] Fehler bei {img_path.name}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Konvertiere Bilder in .pt-Tensors zur Trainingsbeschleunigung.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Ordner mit Originalbildern")
    parser.add_argument("--output-dir", type=Path, required=True, help="Zielordner für .pt-Dateien")
    parser.add_argument("--overwrite", action="store_true", help="Existierende .pt-Dateien überschreiben")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_images(args.input_dir, args.output_dir, args.overwrite)
