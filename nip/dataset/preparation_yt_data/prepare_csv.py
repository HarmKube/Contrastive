import os
import pandas as pd
import imagehash
from PIL import Image
from collections import Counter, defaultdict
from itertools import combinations
import xml.etree.ElementTree as ET
import time
from tqdm.auto import tqdm

# Paths
IMAGE_DIR = "images"
ANNOTATION_DIR = "annotations"
CSV_PATH = "dataset.csv"

# === STEP 1: Generate Dataset CSV ===
def generate_csv(image_dir, annotation_dir, output_csv):
    print("Scanning image directory...")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    data = []
    with tqdm(total=len(image_files), desc="Building CSV dataset") as pbar:
        for img_file in image_files:
            image_path = os.path.join(image_dir, img_file)
            annotation_file = img_file.replace('.jpg', '.xml')
            annotation_path = os.path.join(annotation_dir, annotation_file)
            data.append({
                "original_image_name": img_file,
                "image_path": image_path,
                "annotation_path": annotation_path if os.path.exists(annotation_path) else None
            })
            pbar.update(1)
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"CSV saved to {output_csv} | Images processed: {len(df)}")
    return df

# === STEP 2: Compute pHash ===
def compute_phash(df, image_column='image_path'):
    print("Computing perceptual hashes...")
    hashes = []
    for path in tqdm(df[image_column], desc="Hashing images"):
        try:
            img = Image.open(path)
            phash = str(imagehash.phash(img, hash_size=10))
            hashes.append(phash)
        except Exception as e:
            print(f"Error hashing {path}: {e}")
            hashes.append(None)
    df['pHash_new'] = hashes
    print(f"pHash computed | Missing hashes: {df['pHash_new'].isna().sum()}")
    return df

# === STEP 3: Parse Annotations ===
def parse_annotation(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = [obj.find('name').text for obj in root.findall('object')]
        return sorted(objects)
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return []

def add_annotations(df):
    print("Processing annotations...")
    object_counts = []
    num_objects = []
    with tqdm(total=len(df), desc="Extracting objects") as pbar:
        for xml_path in df['annotation_path']:
            objects = parse_annotation(xml_path) if xml_path else []
            object_counts.append(Counter(objects))
            num_objects.append(len(objects))
            pbar.update(1)
    df['object_counts'] = object_counts
    df['num_objects'] = num_objects
    print(f"Annotations processed | Images with annotations: {df['annotation_path'].notna().sum()}")
    return df

# === STEP 4: Check Consistency in Duplicate pHash Groups ===
def compare_annotations(a, b):
    return a.items() == b.items()

def group_consistency_check(df, phash_column='pHash_new'):
    print("Checking duplicate group consistency...")
    consistent, inconsistent = [], []
    groups = df.groupby(phash_column)
    for phash, group in tqdm(groups, total=len(groups), desc="Analyzing groups"):
        if len(group) < 2:
            continue
        annotations = group['object_counts'].tolist()
        first = annotations[0]
        if all(compare_annotations(first, a) for a in annotations[1:]):
            consistent.append((phash, group))
        else:
            inconsistent.append((phash, group))
    print(f"Group check complete | Consistent: {len(consistent)} | Inconsistent: {len(inconsistent)}")
    return consistent, inconsistent

# === STEP 5: Hamming Distance Grouping ===
def hamming_dist(hash1, hash2):
    return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')

def build_similarity_groups(df, threshold=3):
    print("Building similarity groups...")
    df['keep'] = True
    df_valid = df[df['keep']]
    group_id = 0
    group_labels = pd.Series(index=df.index, dtype=object)
    indices = df_valid.index.tolist()
    total_pairs = len(indices) * (len(indices) - 1) // 2

    print(f"Comparing {total_pairs:,} image pairs...")
    start_time = time.time()
    edges = []
    for i, j in tqdm(combinations(indices, 2), total=total_pairs, desc="Measuring distances"):
        h1, h2 = df.loc[i, 'pHash_new'], df.loc[j, 'pHash_new']
        if pd.notna(h1) and pd.notna(h2) and hamming_dist(h1, h2) <= threshold:
            edges.append((i, j))
    print(f"Distance calculation completed in {time.time()-start_time:.2f}s | Found {len(edges)} similar pairs")

    parent = {i: i for i in indices}
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj
    with tqdm(total=len(edges), desc="Clustering images") as pbar:
        for i, j in edges:
            union(i, j)
            pbar.update(1)
    clusters = defaultdict(list)
    for i in indices:
        clusters[find(i)].append(i)
    with tqdm(total=len(clusters), desc="Labeling groups") as pbar:
        for members in clusters.values():
            for idx in members:
                group_labels[idx] = f"sim_{group_id}"
            group_id += 1
            pbar.update(1)
    df['sim_group'] = group_labels
    print(f"Similarity groups built | Total groups: {group_id}")
    return df

# === STEP 6: Classify Similarity Groups ===
def classify_sim_groups(df):
    print("Classifying similarity groups...")
    consistent, inconsistent = [], []
    groups = df['sim_group'].value_counts()
    multi_image_groups = groups[groups > 1].index
    with tqdm(total=len(multi_image_groups), desc="Processing groups") as pbar:
        for gid in multi_image_groups:
            group = df[df['sim_group'] == gid]
            annotations = group['object_counts'].tolist()
            first = annotations[0]
            if all(compare_annotations(first, a) for a in annotations[1:]):
                consistent.append((gid, group))
            else:
                inconsistent.append((gid, group))
            pbar.update(1)
    print(f"Classification complete | Consistent: {len(consistent)} | Inconsistent: {len(inconsistent)}")
    return consistent, inconsistent

# === MAIN FUNCTION ===
def main():
    if os.path.exists(CSV_PATH):
        print(f"Using existing CSV at {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
    else:
        print(f"CSV not found at {CSV_PATH}. Generating new one...")
        df = generate_csv(IMAGE_DIR, ANNOTATION_DIR, CSV_PATH)

    df = compute_phash(df)
    df = add_annotations(df)

    consistent, inconsistent = group_consistency_check(df)

    df['keep'] = True
    for phash, group in consistent:
        df.loc[group.index[1:], 'keep'] = False

    df = build_similarity_groups(df)
    consistent_sim, inconsistent_sim = classify_sim_groups(df)
    for gid, group in consistent_sim:
        df.loc[group.index[1:], 'keep'] = False

    print("Final keep distribution:")
    print(df['keep'].value_counts())
    #df.to_csv("prepared_dataset.csv", index=False)
    print("Processed dataset saved to prepared_dataset.csv")

    
if __name__ == "__main__":
    main()