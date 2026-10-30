import os
import pandas as pd


CSV_PATH = "prepared_dataset.csv"   # Input CSV
IMAGE_COLUMN = "image_path"          # Column with image paths
KEEP_COLUMN = "keep"                 # Column indicating if the image should be kept

def remove_unkept_images(csv_path, image_column, keep_column):
    df = pd.read_csv(csv_path)

    # Check required columns
    if image_column not in df.columns or keep_column not in df.columns:
        raise ValueError(f"CSV must contain '{image_column}' and '{keep_column}' columns.")

    # Select images to delete
    to_delete = df[df[keep_column] == False][image_column]

    print(f"Found {len(to_delete)} images to delete.")
    delete = input("Do you want to delete these images? (yes/no): ").strip().lower()
    if delete != 'yes':
        print("Deletion cancelled.")
        return
    # Remove each image
    for img_path in to_delete:
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        try:
            if os.path.exists(img_path) and os.path.exists(label_path):
                os.remove(img_path)
                os.remove(label_path)
                print(f"Deleted: {img_path} and {label_path}")
            else:
                print(f"File not found (skipped): {img_path}")
        except Exception as e:
            print(f"Error deleting {img_path}: {e}")

if __name__ == "__main__":
    remove_unkept_images(CSV_PATH, IMAGE_COLUMN, KEEP_COLUMN)
