import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import imagehash
from PIL import Image
import pickle
from tqdm import tqdm

def compute_image_hashes(directory, hash_func=imagehash.phash, hash_size=8, use_cache=True):
    """
    Computes perceptual hashes for all images or loads already computed hashes.
    
    Args:
        directory: Path to the image directory
        hash_func: Hash function (phash, dhash, etc.)
        hash_size: Size of the hash
        use_cache: Whether to use cache
        
    Returns:
        Dictionary with image paths as keys and hash objects as values
    """
    # Create name for the cache file
    hash_name = hash_func.__name__
    cache_dir = "hash_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{hash_name}_{hash_size}.pkl")
    
    # Check if a cache file exists
    if use_cache and os.path.exists(cache_file):
        print(f"Loading stored hashes from {cache_file}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Collect all image paths
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(Path(directory).glob(ext)))
    
    print(f"Found {len(image_files)} images in total.")
    
    # Compute hashes for all images
    hashes = {}
    for img_path in tqdm(image_files, desc=f"Computing {hash_name} hashes (size {hash_size})"):
        try:
            img = Image.open(img_path)
            h = hash_func(img, hash_size=hash_size)
            hashes[img_path] = h
        except Exception as e:
            print(f"Error with {img_path}: {e}")
    
    # Save computed hashes
    if use_cache:
        print(f"Saving hashes to {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump(hashes, f)
    
    return hashes

def find_similar_images(directory, hash_func=imagehash.phash, hash_size=8, threshold=5, use_cache=True):
    """
    Finds similar images in the specified directory.
    
    Args:
        directory: Path to the image directory
        hash_func: Hash function (phash, dhash, etc.)
        hash_size: Size of the hash
        threshold: Maximum Hamming distance for similarity
        use_cache: Whether to use hash cache
        
    Returns:
        List of groups of similar images
    """
    # Compute hashes or load from cache
    hashes = compute_image_hashes(directory, hash_func, hash_size, use_cache)
    
    # Create name for the similarity groups cache file
    hash_name = hash_func.__name__
    cache_dir = "hash_cache"
    groups_cache_file = os.path.join(cache_dir, f"{hash_name}_{hash_size}_threshold_{threshold}_groups.pkl")
    
    # Check if already computed similarity groups exist
    if use_cache and os.path.exists(groups_cache_file):
        print(f"Loading stored similarity groups from {groups_cache_file}...")
        with open(groups_cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Extract image paths from hash keys
    image_files = list(hashes.keys())
    
    # Group similar images
    similar_groups = []
    processed = set()
    
    print(f"Searching for similar images (threshold: {threshold})...")
    for path1 in tqdm(image_files):
        if path1 in processed:
            continue
            
        current_group = [path1]
        processed.add(path1)
        
        for path2 in image_files:
            if path1 == path2 or path2 in processed:
                continue
                
            # Compute Hamming distance
            distance = hashes[path1] - hashes[path2]
            if distance <= threshold:
                current_group.append(path2)
                processed.add(path2)
        # Only save groups with multiple images
        if len(current_group) > 1:
            similar_groups.append(current_group)
    
    # Save computed similarity groups
    if use_cache:
        print(f"Saving similarity groups to {groups_cache_file}...")
        with open(groups_cache_file, 'wb') as f:
            pickle.dump(similar_groups, f)
    
    return similar_groups


def parse_annotation(xml_path):
    """Parses XML annotations and returns a sorted object list"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            # for now we do not care about where the objects are
            bndbox = obj.find('bndbox')
            bbox = {
                'xmin': int(bndbox.find('xmin').text),
                'ymin': int(bndbox.find('ymin').text),
                'xmax': int(bndbox.find('xmax').text),
                'ymax': int(bndbox.find('ymax').text)
            }
            objects.append(name)  # Tuple for better comparability
        
        # Sort by name
        return sorted(objects, key=lambda x: (x[0]))
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None

def compare_annotations(ann1, ann2):
    """Compares two annotations for content equality"""
    if ann1 is None or ann2 is None:
        return False
    
    # for now we say if one of the pictures contains a subset of objects of the other we are good
    # -> FALSE = both pictures contain something the other one does not contain -> see different things
    if set(ann1).issubset(set(ann2)) or set(ann2).issubset(set(ann1)):
        return True
            
    return False

def process_group_with_annotations(group, image_dir, annotation_dir):
    """Processes an image group with annotation check"""
    # Collect XML paths
    group_annotations = {}
    for img_path in group:
        img_name = os.path.basename(str(img_path))
        xml_path = os.path.join(annotation_dir, f"{Path(img_name).stem}.xml")
        group_annotations[img_path] = parse_annotation(xml_path) if os.path.exists(xml_path) else None

    # Find reference annotation
    ref_ann = next((ann for ann in group_annotations.values() if ann is not None), None)
    if not ref_ann:
        return {"consistent": group, "conflicting": []}

    # Split groups
    consistent = []
    conflicting = []
    for img_path, ann in group_annotations.items():
        if compare_annotations(ann, ref_ann):
            consistent.append(img_path)
        else:
            conflicting.append(img_path)
    return {"consistent": consistent, "conflicting": conflicting}

def save_annotated_groups(groups, image_dir, annotation_dir, output_base="annotated_groups"):
    """Saves groups with annotation check"""
    consistent_dir = os.path.join(output_base, "consistent")
    conflicting_dir = os.path.join(output_base, "conflicting")
    for group_idx, group in enumerate(groups, 1):
        processed = process_group_with_annotations(group, image_dir, annotation_dir)
        
        # Save consistent group
        if processed["consistent"]:
            group_dir = os.path.join(consistent_dir, f"group_{group_idx:03d}")
            os.makedirs(group_dir, exist_ok=True)
            for img_path in processed["consistent"]:
                copy_with_annotation(img_path, group_dir, annotation_dir)
        
        # Save conflicting images
        if processed["conflicting"]:
            conflict_dir = os.path.join(conflicting_dir, f"group_{group_idx:03d}")
            os.makedirs(conflict_dir, exist_ok=True)
            for img_path in processed["conflicting"]:
                copy_with_annotation(img_path, conflict_dir, annotation_dir)

def copy_with_annotation(img_path, target_dir, annotation_dir):
    """Copies image and corresponding annotation"""
    # Copy image
    img_name = os.path.basename(str(img_path))
    shutil.copy2(str(img_path), os.path.join(target_dir, img_name))
    
    # Copy annotation
    xml_name = f"{Path(img_name).stem}.xml"
    xml_src = os.path.join(annotation_dir, xml_name)
    if os.path.exists(xml_src):
        shutil.copy2(xml_src, os.path.join(target_dir, xml_name))

# Modified analyze_duplicates function with annotation support
def analyze_duplicates_with_annotations(
    image_dir="dataset/JPEGImages",
    annotation_dir="dataset/Annotations",
    hash_type="phash",
    hash_size=8,
    threshold=5,
    use_cache=True,
    output_base="annotated_groups"
):
    # Existing hashing and grouping logic
    hash_func = getattr(imagehash, hash_type)
    groups = find_similar_images(image_dir, hash_func, hash_size, threshold, use_cache)
    
    # New annotation processing
    save_annotated_groups(
        groups,
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        output_base=output_base
    )
    # Print statistics
    total = sum(len(g) for g in groups)
    print(f"\nFinal groups saved in {output_base}/")
    print(f"Total similar images found: {total}")

# Example usage
if __name__ == "__main__":
    analyze_duplicates_with_annotations(
        image_dir="/mydata/GERALD/dataset/JPEGImages",
        annotation_dir="/mydata/GERALD/dataset/Annotations",
        hash_type="phash",
        hash_size=10,
        threshold=3,
        output_base="annotated_duplicates"
    )
