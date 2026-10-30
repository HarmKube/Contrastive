import os
import pandas as pd
import shutil

"""
This script was used to reorganize the GERALD dataset, which was copied from /mydata/GERALDsign, based on the splits defined in a CSV file.
This script organizesthe image and label datasets based on a CSV file (`df_splits_final.csv`) that defines the correct split ('train', 'test', or 'valid') for each image.
"""
import os
import pandas as pd
import shutil

csv_path = 'df_splits.csv'
images_root = 'images'
labels_root = 'labels'
splits = ['train', 'test', 'valid']


df = pd.read_csv(csv_path)
df['image'] = df['image'].str.strip().str.replace('.JPG', '.jpg').str.replace('.jpeg', '.jpg', regex=False)

allowed_images = set(df['image'])
allowed_labels = set(img.replace('.jpg', '.txt') for img in allowed_images)
missing_images = []
missing_labels = []

# Move files listed in the CSV
for _, row in df.iterrows():
    image_name = row['image']
    label_name = image_name.replace('.jpg', '.txt')
    split = row['split']

    img_src = None
    lbl_src = None
    img_found = False
    lbl_found = False

    # Try to find the file in any split folder
    for part in splits:
        img_candidate = os.path.join(images_root, part, image_name)
        lbl_candidate = os.path.join(labels_root, part, label_name)
        if os.path.exists(img_candidate):
            img_src = img_candidate
            img_found = True
        if os.path.exists(lbl_candidate):
            lbl_src = lbl_candidate
            lbl_found = True

    # Destination paths
    img_dst = os.path.join(images_root, split, image_name)
    lbl_dst = os.path.join(labels_root, split, label_name)

    # Move image
    if img_found:
        if img_src != img_dst:
            shutil.move(img_src, img_dst)
            print(f"Moved image: {img_src} → {img_dst}")
    else:
        print(f"[Missing Image] {image_name}")
        missing_images.append(image_name)

    # Move label
    if lbl_found:
        if lbl_src != lbl_dst:
            shutil.move(lbl_src, lbl_dst)
            print(f"Moved label: {lbl_src} → {lbl_dst}")
    else:
        print(f"[Missing Label] {label_name}")
        missing_labels.append(label_name)

# Step 2: Delete any files not listed in the CSV
for split in splits:
    image_dir = os.path.join(images_root, split)
    label_dir = os.path.join(labels_root, split)

    # Delete extra images
    for f in os.listdir(image_dir):
        if f.endswith('.jpg') and f not in allowed_images:
            os.remove(os.path.join(image_dir, f))
            print(f"[Deleted Extra Image] {f}")

    # Delete extra labels
    for f in os.listdir(label_dir):
        if f.endswith('.txt') and f not in allowed_labels:
            os.remove(os.path.join(label_dir, f))
            print(f"[Deleted Extra Label] {f}")

print(f"Total files listed in CSV: {len(df)}")
print(f"Missing images: {len(missing_images)}")
print(f"Missing labels: {len(missing_labels)}")

log_path = 'missing_files_log.txt'
with open(log_path, 'w') as f:
    f.write("Missing Images:\n")
    f.writelines([img + '\n' for img in missing_images])
    f.write("\nMissing Labels:\n")
    f.writelines([lbl + '\n' for lbl in missing_labels])

print(f"\nMissing file report saved to: {log_path}")