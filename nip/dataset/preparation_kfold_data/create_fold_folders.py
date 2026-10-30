import os
import pandas as pd

""""
This script is executed on the hpc server to create the folder structure for the folds:
e.g. a path looks like this afterwards: GERALDfolds/folds/fold0/test.txt
 => test.txt contains paths to the images and labels for the test set of fold0.
"""

csv_path = 'df_folds.csv'  # can be found in nip/notebooks/kfolds
base_path = '/mydata/GERALDfolds'
image_base_path = os.path.join(base_path, 'original', 'images')
folds_root = os.path.join(base_path, 'folds')

df = pd.read_csv(csv_path)
df = df[df['keep'] == True]

for fold in range(5):
    print(f"Creating Fold {fold}...")

    # Define fold directory
    fold_dir = os.path.join(folds_root, f'fold{fold}')
    os.makedirs(f"./folds/fold{fold}", exist_ok=True)

    # Apply split criteria
    val_df = df[df['fold'] == 2 * fold]
    test_df = df[df['fold'] == 2 * fold + 1]
    train_df = df[(df['fold'] != 2 * fold) & (df['fold'] != 2 * fold + 1)]

    # Helper to write image paths to txt
    def write_paths(df_subset, filename, fold):
        paths = df_subset['image'].apply(lambda img: os.path.join(image_base_path, img))
        with open(os.path.join(f"./folds/fold{fold}", filename), 'w') as f:
            for path in paths:
                f.write(f"{path}\n")

    write_paths(train_df, 'train.txt', fold)
    write_paths(val_df, 'val.txt', fold)
    write_paths(test_df, 'test.txt', fold)

print("Done creating all 5 folds.")
