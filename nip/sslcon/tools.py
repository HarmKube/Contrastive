import torch
import lightly

from ultralytics import YOLO
from torch import nn

from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

import os
from collections import defaultdict


def compute_stats(val_results, path_to_data, output_path):
    """
    Computes statistics of the resulting object of yolo.val() in val_results. This process requires path_to_data
    to also compute statistics like image and instance count. A results.csv is stored in output_path.
    """
    image_extensions = {'.jpg', '.jpeg', '.png'}
    num_images = 0
    num_instances = 0
    num_images_test = 0
    num_instances_test = 0

    class_counts_images = defaultdict(int)
    class_counts = defaultdict(int)
    class_counts_images_test = defaultdict(int)
    class_counts_test = defaultdict(int)

    # Iterate through all the splits to gather information
    for split in ["train", "valid", "test"]:
        image_path = Path(f"{path_to_data}/images/{split}")
        test_dir = Path(f"{path_to_data}/labels/{split}")

        # Iterate over all files in the directory
        for filename in os.listdir(image_path):
            # Check for image files
            if os.path.splitext(filename)[1].lower() in image_extensions:
                num_images += 1
                if split == "test":
                    num_images_test += 1
                label_filename = os.path.splitext(filename)[0] + '.txt'
                label_path = os.path.join(test_dir, label_filename)

                # If label file exists, parse it
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        class_counts_temp = defaultdict(int)
                        for line in f:
                            if line.strip():  # skip empty lines
                                class_id = line.strip().split()[0]

                                if class_id not in class_counts_temp:
                                    class_counts_temp[class_id] = True
                                    class_counts_images[class_id] += 1
                                    if split == "test":
                                        class_counts_images_test[class_id] += 1

                                class_counts[class_id] += 1
                                num_instances += 1
                                if split == "test":
                                    num_instances_test += 1
                                    class_counts_test[class_id] += 1

    # Extract per-class metrics
    class_ids = val_results.names.keys()
    class_names = val_results.names.values()
    precision = val_results.box.p.tolist()
    recall = val_results.box.r.tolist()
    map50 = val_results.box.ap50.tolist()
    map50_95 = val_results.box.ap.tolist()

    data = []

    # Add the average results
    data.append({
        'class_id': -1,
        'class_name': "average",
        'nr_of_images': num_images,
        'nr_of_instances': num_instances,
        'nr_of_images_test': num_images_test,
        'nr_of_instances_test': num_instances_test,
        'precision': val_results.box.p.mean().item(),
        'recall': val_results.box.r.mean().item(),
        'mAP50': val_results.box.ap50.mean().item(),
        'mAP50-95': val_results.box.ap.mean().item(),
    })

    offset = 0
    # Iterate through classes
    for i, name in zip(class_ids, class_names):
        if i in val_results.ap_class_index:
            data.append({
                'class_id': i,
                'class_name': name,
                'nr_of_images': class_counts_images[str(i)],
                'nr_of_instances': class_counts[str(i)],
                'nr_of_images_test': class_counts_images_test[str(i)],
                'nr_of_instances_test': class_counts_test[str(i)],
                'precision': precision[i - offset],
                'recall': recall[i - offset],
                'mAP50': map50[i - offset],
                'mAP50-95': map50_95[i - offset],
            })
        else:
            # Handle case, when no instance in test set
            offset += 1
            data.append({
                'class_id': i,
                'class_name': name,
                'nr_of_images': class_counts_images[str(i)],
                'nr_of_instances': class_counts[str(i)],
                'precision': -1,
                'recall': -1,
                'mAP50': -1,
                'mAP50-95': -1,
            })

    # Save to CSV
    df = pd.DataFrame(data)

    df.to_csv(Path(f"{output_path}/val_results.csv"), index=False)
