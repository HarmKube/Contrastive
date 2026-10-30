import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os


def save_kfolds_results(experiment, output_path="out/kfolds", metrics=["mAP50", "mAP50-95", "precision", "recall"]):

    """
    Saves the specified metrics from the k-folds validation results into a CSV file into the kfolds folder. Please push these results to git.
    """

    metrics = [m.strip() for m in metrics]
    folds = ["GERALD_0", "GERALD_1", "GERALD_2", "GERALD_3", "GERALD_4"]
    all_results = {}

    for metric in metrics:
        results = []
        for folder in folds:
            print(f"Processing folder: {folder} for metric: {metric}")
            
            script_dir = Path(__file__).resolve().parent.parent.parent
            path = script_dir / output_path / "kfolds" / folder / "VAL/val_results.csv"

            if not path.exists():            
                raise FileNotFoundError(f"File {path} does not exist. There are missing folds! Make sure there are 5 GERALD folders WITH results in the output path")
            
            df = pd.read_csv(path)
            if metric not in df.columns:
                print(f"Metric '{metric}' not found in {path}. Available columns: {df.columns}")
                results.append(None)
                continue

            value = df.loc[0, metric]
            results.append(value)
        
        if all(r is None for r in results):
            print(f"Warning: No valid results collected for metric '{metric}'. Skipping.")
            continue
        
        all_results[metric] = results

    if not all_results:
        raise ValueError("No results collected for any metrics. Check if the metrics exist in the CSV files.")

    # Create DataFrame: rows are metrics, columns are folds
    results_df = pd.DataFrame(all_results, index=folds).T  # Transpose to match desired format
    results_df.columns = folds

    save_results_path = Path(__file__).resolve().parent / "results"
    save_results_path.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(save_results_path / f'{experiment}_results.csv', index=True)
    
# Run this, when all results are collected, to generate boxplots for a metrics over all experiments
def boxplots_stats_all_experiments(folder_path = "results", metric = "mAP50-95"):

    all_data = {}
    
    folder_path = Path(__file__).resolve().parent / folder_path
    # Iterate through all CSV files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath, index_col=0)
            
            if metric in df.index:
                values = df.loc[metric].values
                experiment_name = filename.replace("_results.csv", "")
                all_data[experiment_name] = values
            else:
                print(f"Metric '{metric}' not found in {filename}. Skipping.")

    if not all_data:
        print(f"No data found for metric '{metric}'.")
        return

    plt.figure(figsize=(10, 6))
    plt.boxplot(all_data.values(), tick_labels=all_data.keys())
    plt.ylabel(metric)
    plt.title(f"Boxplots of '{metric}' Across Experiments")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(folder_path / f"{metric}_boxplot", dpi=300)
    #plt.show()
    plt.close()


if __name__ == "__main__":
     
    # TODO 1: change the experiment name and the output path according to the experiment you are running
    # TODO 2: push the resulting csv that is generated in the directory of this script to git
    
    save_kfolds_results(experiment="test", output_path="out/kfolds")     

    # For later, when the results of all experiments are collected run the following function to generate a boxplot over all experiments:
    boxplots_stats_all_experiments(metric="mAP50-95")

