"""
Resize images to a fixed size (no aspect-ratio preservation, kein Padding)
and copy the YOLO labels 1-zu-1, weil die relativen Koordinaten unverändert
gültig bleiben.

Example
-------
resize_and_copy_labels(
    image_dir        = ".../images/valid",
    label_dir        = ".../labels/valid",
    output_image_dir = ".../resized/images/valid",
    output_label_dir = ".../resized/labels/valid",
    target_size      = (640, 640),
)
"""
from pathlib import Path
from typing import Tuple

import os
from PIL import Image
from tqdm import tqdm


def resize_and_copy_labels(
    image_dir: str,
    label_dir: str,
    output_image_dir: str,
    output_label_dir: str,
    target_size: Tuple[int, int] = (640, 640),
) -> None:
    """Resize all images in *image_dir* to *target_size* (WxH) and copy the
    corresponding YOLO label files unchanged to the output directory.

    Parameters
    ----------
    image_dir : str
        Path to input images.
    label_dir : str
        Path to YOLO text files with the same basenames.
    output_image_dir : str
        Destination for the resized images.
    output_label_dir : str
        Destination for the (unchanged) label files.
    target_size : (int, int), optional
        Output resolution (width, height), by default (640, 640).

    Notes
    -----
    * Für ein reines Resize ohne Letterbox-/Padding-Logik **muss** man die
      Bounding-Boxes **nicht** anpassen.  
    * Wenn du stattdessen Letterboxing verwendest, brauchst du eine andere
      Transformation (Skalieren + Shifts).  
      → Melde dich, falls du das brauchst.
    """
    # Zielordner erstellen
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Alle Bilddateien durchlaufen
    for img_path in tqdm(
        Path(image_dir).iterdir(), desc="Resizing (labels werden kopiert)"
    ):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        # Bild laden, auf Zielgröße bringen und speichern
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize(target_size, Image.BILINEAR)
        img_resized.save(Path(output_image_dir, img_path.name))

        # Label-Datei 1-zu-1 kopieren (wenn vorhanden)
        label_file = Path(label_dir, img_path.stem + ".txt")
        if label_file.exists():
            dest_label_file = Path(output_label_dir, label_file.name)
            dest_label_file.write_text(label_file.read_text(encoding="utf-8"))


# Beispielaufruf --------------------------------------------------------------
if __name__ == "__main__":
    resize_and_copy_labels(
        image_dir="/mydata/GERALDfinal/images/valid",
        label_dir="/mydata/GERALDfinal/labels/valid",
        output_image_dir="/mydata/GERALDfinalresized/images/valid",
        output_label_dir="/mydata/GERALDfinalresized/labels/valid",
        target_size=(640, 640),
    )
