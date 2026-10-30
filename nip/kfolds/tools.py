from collections import defaultdict
from pathlib import Path
import os
import pandas as pd
import yaml


def compute_stats(val_results, yaml_data_dict, output_path):
    image_extensions = {'.jpg', '.jpeg', '.png'}

    with open(yaml_data_dict, 'r') as f:
        data_config = yaml.safe_load(f)
    yaml_data_dict = data_config

    split_files = {
        "train": yaml_data_dict.get("train"),
        "val": yaml_data_dict.get("val"),
        "test": yaml_data_dict.get("test"),
    }

    num_images = 0
    num_instances = 0
    num_images_test = 0
    num_instances_test = 0

    class_counts_images = defaultdict(int)
    class_counts = defaultdict(int)
    class_counts_images_test = defaultdict(int)
    class_counts_test = defaultdict(int)

    for split, file in split_files.items():
        if not file or not os.path.exists(file):
            continue
        with open(file, "r") as f:
            lines = f.readlines()

        for line in lines:
            img_path = Path(line.strip())
            if not img_path.exists():
                continue

            label_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
            if label_path.exists():
                num_images += 1
                if split == "test":
                    num_images_test += 1
                with open(label_path, "r") as lf:
                    class_counts_temp = set()
                    for lbl_line in lf:
                        if lbl_line.strip():
                            class_id = lbl_line.strip().split()[0]
                            class_counts_temp.add(class_id)
                            class_counts[class_id] += 1
                            num_instances += 1
                            if split == "test":
                                class_counts_test[class_id] += 1
                                num_instances_test += 1
                    for class_id in class_counts_temp:
                        class_counts_images[class_id] += 1
                        if split == "test":
                            class_counts_images_test[class_id] += 1

    # Collect metrics from val_results
    class_ids = val_results.names.keys()
    class_names = val_results.names.values()
    precision = val_results.box.p.tolist()
    recall = val_results.box.r.tolist()
    map50 = val_results.box.ap50.tolist()
    map50_95 = val_results.box.ap.tolist()

    data = []

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

    df = pd.DataFrame(data)
    df.to_csv(Path(output_path) / "val_results.csv", index=False)

# add main function
def main():
    compute_stats()