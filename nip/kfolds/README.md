
# K-Fold Training Pipeline for YOLO

* 5-fold cross-validation
* Auto-resumes from checkpoints
* Per-fold YAML generation
* Validation + stats computation
* Frequency analysis on test results
* Safe restartable execution

#
## Dataset YAML Format

The `gerald_folds.yaml` file should look like this:

```yaml
# Every path should be from the perspective of the container (the bind folder)
path: /mydata/GERALDfolds/original  # Folder containing the actual images

train: /mydata/GERALDfolds/folds/fold0/train.txt  # File listing training image paths
val:   /mydata/GERALDfolds/folds/fold0/val.txt    # File listing validation image paths
test:  /mydata/GERALDfolds/folds/fold0/test.txt   # File listing test image paths
```

* The `fold0` portion in the paths is dynamically replaced with `fold{i}` during each fold iteration.
* The files listed in `train.txt`, `val.txt`, and `test.txt` should contain full paths to the real images (from the container's point of view).

## Output Structure

Each fold's results are saved under:

```
/out/kfolds/
  ├── kfolds/
  │    ├── GERALD_0/
  │    │    ├── weights/
  │    │    ├── VAL/
  │    │    ├── results.csv
  │    │    ├── args.yaml
  │    │    └── ...
  ├── gerald_fold_0.yaml
  ├── gerald_fold_1.yaml
  ...
```

## Outputs

* `results.csv`: YOLO training log
* `VAL/`: Validation outputs and stats
* `val_done.txt`: Marker indicating validation completion
* Cleaned and corrected `args.yaml` per fold to ensure reproducibility

## Notes

* Automatically resumes training if a checkpoint and `args.yaml` exist
* Fixes any incorrect values in `args.yaml` for `project`, `name`, or `save_dir`
* Re-trains and re-validates only if results are missing or incomplete
* A pretrained YOLO checkpoint can be provided via the `yolo_ckpt` parameter