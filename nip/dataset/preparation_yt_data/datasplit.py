import os
import random
import shutil

# Verzeichnisse
images_dir = "images"
labels_dir = "labels"
train_images_dir = "images/train"
val_images_dir = "images/val"
train_labels_dir = "labels/train"
val_labels_dir = "labels/val"

# Zielverzeichnisse anlegen
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Alle Bilddateien finden
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]


# Shuffle und Split
random.seed(42)
random.shuffle(image_files)
val_count = int(len(image_files) * 0.05)
val_files = set(image_files[:val_count])
train_files = set(image_files[val_count:])

def copy_files(img_set, img_dst, label_dst):
    for img_file in img_set:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        shutil.move(os.path.join(images_dir, img_file), os.path.join(img_dst, img_file))
        shutil.move(os.path.join(labels_dir, label_file), os.path.join(label_dst, label_file))

copy_files(train_files, train_images_dir, train_labels_dir)
copy_files(val_files, val_images_dir, val_labels_dir)

print(f"Train: {len(train_files)} images")
print(f"Val: {len(val_files)} images")