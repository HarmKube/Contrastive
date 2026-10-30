# CLI Usage
This module provides a command-line interface entry point via `nip.py`. Invoke it  from the package root using:

```powershell
python nip/nip.py <action> [options]
```

The CLI offers a easy-to-use interface for many high-level functions of this repository. However, not all functions
might be integrated and some scripts run on their own in the respective packages. The requirements are the same for 
both, though, and can be found in the respective [requirements.txt](../env/requirements.txt) for building the container.

# Available Actions
All actions support long-form (`--action`) and short-form flags as listed.

## cli-test
Test the CLI setup.

Usage:
```powershell
python nip/nip.py --cli-test
```

## Help
The help can be displayed for every action and the complete CLI via:
```powershell
python nip/nip.py -h

python nip/nip.py --action -h
```

## Config
By default the cli loads the `./confgi/default.yaml` file.
with the flag `-c` a different config file can be specified.

## Main Pre-Training pipeline
### benchmark (`-b`, `--benchmark`)
Run a performance benchmark on a model. FPS-Numbers for the different export formats are shown.
- Required:
  - `--model PATH` : path to the model file
  - `--data PATH`  : path to the dataset config
- Optional:
  - `--imgsz INT`  : image size
  - `--device STR` : compute device

Benchmark example:
```powershell
python nip.py -b --model models/yolov8s.pt --data config/default.yaml --imgsz 640
```

### train (`-t`, `--train`)
Train a model using self-supervised or fine-tuning frameworks.
- Required:
  - `--data PATH` : The configuration-file for GERALD-training
- Optional:
  - `--imgsz INT`: Image size for training
  - `--output PATH` : Path the YOLO evaluation is written to.
  - `--ft_coco`: Specify if you want to pre-train on COCO manually (dataset needs to be downloaded).       
  - `--ssl_mode MODE`: e.g., `YOLOMoCo`, `yolo_detco`, `lightly`, `generic`)
  - `--local_data PATH`: Internal flag for testing.
  - `--moco_ckpt PATH` : Continue training on previous lightning checkpoint for MoCo.
  - `--yolo_ckpt PATH` : Continue from YOLO checkpoint.
  - `--lightly_ckpt`  : Continue training on previous lightly checkpoint for other methods.

Training example:
```powershell
python nip.py -t --data data/gerald_final.yaml --ssl_mode YOLOMoCo --output out/
```

### kfolds (`-kf`, `--kfolds`)
Run k-fold cross-validation training.
- Required:
  - `--data PATH`: Path to `gerald_folds.yaml`. One is provided in [data](../data).
- Optional:
  - `--imgsz INT`: (see above)
  - `--output PATH`: (see above)
  - `--yolo_ckpt PATH`: (see above)
  - `--kfold_iter INT`: Fold from which to continue training.

### validate (`-v`, `--validate`)
Run a validation run on a YOLO model.
- Required:
  - `--output PATH`: Specify path to `GERALD` folder where YOLO output its training results.
  - `--data PATH`: (see above)
- Optional:
  - `--imgsz INT`: (see above)
  - `--device STR`: Specify GPU-device.
  - `--kfolds_val`: Specifies that validation is run k-Fold Validation Fold

Example
```powershell
# On output of normal GERALD fine-tuning
python /mysrc/nip/nip.py -v --data /mysrc/data/gerald_final.yaml --output /mydata/GERALD

# On Fold with k-Fold CV
python /mysrc/nip/nip.py -v --data /mydata/kfolds/gerald_fold_3.yaml --output /mydata/kfolds/kfolds/GERALD_3 -kv
```

## Further utilities
### video-dataset (`-vd`, `--video-dataset`)
Extract a frame dataset from a video or YouTube URL.
- Required:
  - `--source URL|PATH`
  - `--output PATH`
- Optional:
  - `--interval SEC`
  - `--resolution STR`
  - `--output_filename NAME`

### parse_playlist (`-pp`, `--parse_playlist`)
Download and list video URLs from a YouTube playlist.
- Required:
  - `--source URL`
  - `--output PATH`

### create-labels (`-cl`, `--create-labels`)
Generate label files using a YOLO model.
- Required:
  - `--data PATH`
  - `--model PATH`
- Optional:
  - `--output PATH`
  - `--confidence_threshold FLOAT`
  - `--device STR`
  - `--batch_size INT`

### visualize-annotation (`-va`, `--visualize-annotation`)
Render an XML annotation to an image.
- Required:
  - `--xml_path PATH`
- Optional:
  - `--output PATH`

### analyse-dataset (`-ad`, `--analyse-dataset`)
Run dataset analysis scripts.
- Required:
  - `--data PATH`
- Optional:
  - `--split STR`

### compare-analysis (`-ca`, `--compare-analysis`)
Compare two dataset analysis outputs.
- Required:
  - `--file_1 PATH`
  - `--file_2 PATH`

### interpret-global (`-ig`, `--interpret-global`)
Generate global model interpretations using Grad-CAM.
- Required:
  - `--model PATH`
  - `--output PATH`
  - `--data PATH`
  - `--indices LIST`

### interpret-local (`-il`, `--interpret-local`)
Generate local model interpretations using Grad-CAM.
- Required:
  - `--model PATH`
  - `--output PATH`
  - `--data PATH`
  - `--indices LIST`


