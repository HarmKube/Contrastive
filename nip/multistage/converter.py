import os
from pathlib import Path
from collections import Counter

from GroupAssign import GroupAssign as ga

if __name__ == '__main__':
    """
    Quick script to generate labels for Generic pre-training approach.
    Converts labels in labels_dir (see below) to new labels according to assignment in GroupAssign.py.
    Change datasets_path according to the root-folder that you are working in.
    """
    datasets_path = Path("/media/panda/T7/datasets")

    labels_dir = datasets_path / "GERALDfinal/labels"
    output_labels_dir = datasets_path / "GERALDGenericFinal/labels"
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Build dictionary: old_index -> new_index
    old_index_to_new_index = {}
    for idx, name in enumerate(ga.object_names):
        cat = ga.signal_category.get(name)
        if cat is None:
            raise KeyError(f"Name '{name}' (index {idx}) not in original_name_to_category mapping")
        new_idx = ga.category_to_new_index[cat]
        old_index_to_new_index[idx] = new_idx

    # Remapping the labels
    for label_path in labels_dir.rglob("*.txt"):
        relative = label_path.relative_to(labels_dir)
        out_path = output_labels_dir / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines_out = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                old_cls = int(parts[0])
                if old_cls not in old_index_to_new_index:
                    continue  # or handle differently
                new_cls = old_index_to_new_index[old_cls]
                lines_out.append(f"{new_cls} " + " ".join(parts[1:]))
        with open(out_path, "w") as f:
            for l in lines_out:
                f.write(l + "\n")

    # Summary
    counter = Counter()
    for txt in output_labels_dir.rglob("*.txt"):
        for line in open(txt):
            cls = int(line.split()[0])
            counter[cls] += 1
    print("New-class distribution:", counter)


    # Print sample mappings for sanity-check
    for old_idx in [0, 1, 14, 30]:
        name = ga.object_names[old_idx]
        new_idx = old_index_to_new_index[old_idx]
        # find category name from category_to_new_index:
        cat = next(k for k,v in ga.category_to_new_index.items() if v == new_idx)
        print(f"Old idx {old_idx}: '{name}' → new idx {new_idx} ('{cat}')")