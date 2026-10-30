from ultralytics import YOLO
from pathlib import Path
import yaml
import os
import pandas as pd
import shutil

from config.global_config import config as cfg
from kfolds.tools import compute_stats
from benchmarks.val import analyze_frequencies


"""
The gerald_folds.yaml file should look like this:

    # every path from the view of the container (the bind folder)
    path: /mydata/GERALDfolds/original # folder with the real images
    train: /mydata/GERALDfolds/folds/fold0/train.txt # paths to the real images that should be used for training, e.g: /mydata/GERALDfolds/original/images/Aachen-Rheydt.mp4#t=1481.8.jpg
    val: /mydata/GERALDfolds/folds/fold0/val.txt # paths to the real images that should be used for validation 
    test: /mydata/GERALDfolds/folds/fold0/test.txt # paths to the real images that should be used for testing 

The fold number (here fold0) in the paths will be replaced by the current fold number during training.
The paths in each of the splits (train, val, test) should contain the paths to the real images (again from the view of the container (the bind folder)).


"""


def kfolds(data="gerald_folds.yaml", imgsz=1920, output="/out/kfolds", yolo_ckpt=None, kfold_iter=0):
    """
    K-Fold CV is performed with the yolo_ckpt parameter on data. If no yolo_ckpt is specified, it will use yolo_ckpt
    from config. kfold_iter can allow to continue from a fold when there is still data in output folder.
    """
    kfold_iter = int(kfold_iter)
    data = Path(data)
    output = Path(output)
    num_folds = 5

    with open(data, "r") as f:
        base_data = yaml.safe_load(f)

    sample_train_path = Path(base_data["train"])
    folds_base = sample_train_path.parent.parent

    """
    # Cleanup: remove unexpected GERALD_* folders
    expected = {f"GERALD_{i}" for i in range(num_folds)}
    for d in (output / "kfolds").glob("GERALD_*"):
        if d.name not in expected:
            print(f"[CLEANUP] Removing unexpected folder: {d}")
            shutil.rmtree(d)
    """

    if not os.path.exists(output):
        os.makedirs(output)

    for i in range(num_folds):
        if i < kfold_iter:
            continue
        fold_dir = output / "kfolds" / f"GERALD_{i}"
        weights_dir = fold_dir / "weights"
        val_dir = fold_dir / "VAL"
        tmp_yaml_path = output / f"gerald_fold_{i}.yaml"
        log_file = fold_dir / "results.csv"
        val_marker = val_dir / "val_done.txt"
        args_path = fold_dir / "args.yaml"
        last_ckpt = weights_dir / "last.pt"

        # Check if training is already done
        training_done = False
        if log_file.exists():
            try:
                df = pd.read_csv(log_file)
                training_done = len(df) >= cfg.kfolds_config.yolo_gerald_epochs
            except Exception as e:
                print(f"[WARN] Could not read log file for fold {i}: {e}")

        # Create YAML for current fold
        dynamic_data = base_data.copy()
        dynamic_data["train"] = str(folds_base / f"fold{i}/train.txt")
        dynamic_data["val"] = str(folds_base / f"fold{i}/val.txt")
        dynamic_data["test"] = str(folds_base / f"fold{i}/test.txt")
        dynamic_data.pop(":path", None)

        with open(tmp_yaml_path, "w") as f:
            yaml.dump(dynamic_data, f)

        # Start/resume training if not already done
        if not training_done:
            if last_ckpt.exists() and args_path.exists():
                print(f"[INFO] Resuming training for fold {i} from checkpoint.")
                yolo = YOLO(str(last_ckpt))
            else:
                if yolo_ckpt:
                    print(f"[INFO] Using pretrained weights, starting fresh training: {yolo_ckpt}")
                    yolo = YOLO(str(yolo_ckpt))
                else:                    
                    print(f"[INFO] Starting fresh training for fold {i}")
                    model_path = cfg.kfolds_config.yolo_version
                    yolo = YOLO(model_path if ".yaml" in model_path else f"{output}/{model_path}")
                    if args_path.exists():
                        print(f"[CLEANUP] Removing stale args.yaml: {args_path}")
                        args_path.unlink()

            # Fix args.yaml if it exists and has wrong project/name/save_dir
            expected_project = str(fold_dir.parent)
            expected_name = fold_dir.name
            expected_save_dir = str(fold_dir)

            if args_path.exists():
                try:
                    with open(args_path, "r") as f:
                        args_data = yaml.safe_load(f)

                    needs_fix = (
                        args_data.get("project") != expected_project or
                        args_data.get("name") != expected_name or
                        args_data.get("save_dir") != expected_save_dir
                    )

                    if needs_fix:
                        print(f"[FIX] Correcting args.yaml for fold {i}")
                        args_data["project"] = expected_project
                        args_data["name"] = expected_name
                        args_data["save_dir"] = expected_save_dir
                        with open(args_path, "w") as f:
                            yaml.dump(args_data, f)
                except Exception as e:
                    print(f"[WARN] Failed to fix args.yaml for fold {i}: {e}")

            # Train
            try:
                print("[INFO] Start training (without plots)")
                yolo.train(
                    data=str(tmp_yaml_path),
                    epochs=cfg.kfolds_config.yolo_gerald_epochs,
                    imgsz=imgsz,
                    batch=cfg.kfolds_config.yolo_batch_size,
                    project=fold_dir.parent,
                    name=fold_dir.name,
                    exist_ok=True,
                    resume=last_ckpt.exists() and args_path.exists(),
                    plots=False
                )
            except Exception as e:
                print(f"[ERROR] Fold {i} training failed: {e}")
                continue
        else:
            print(f"[INFO] Fold {i} already trained — skipping training step.")

        # Validate and compute stats
        if not val_marker.exists():
            try:
                print(f"[INFO] Validating fold {i}...")
                
                yolo = YOLO(f"{fold_dir}/weights/best.pt")

                val_results = yolo.val(
                    model=str(weights_dir / "best.pt"),
                    data=str(tmp_yaml_path),
                    imgsz=imgsz,
                    batch=cfg.kfolds_config.yolo_batch_size,
                    split="test",
                    project=fold_dir,
                    name="VAL",
                    exist_ok=True
                )
                

                print(f"[INFO] Computing stats for fold {i}...")
                compute_stats(val_results, tmp_yaml_path, val_dir)
                print(f"[INFO] Analyzing frequencies for fold {i}...")
                analyze_frequencies(val_dir)

                val_marker.write_text("done")
            except Exception as e:
                print(f"[ERROR] Fold {i} validation/stats failed: {e}")
        else:
            print(f"[INFO] Fold {i} validation/stats already done.")

        # --- Add cleanup to reset distributed training state ---
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
                print(f"[INFO] Destroyed distributed process group after fold {i}.")
        except Exception as e:
            print(f"[WARN] Failed to destroy distributed group after fold {i}: {e}")
        
        # Clean up memory and GPU caches
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()
