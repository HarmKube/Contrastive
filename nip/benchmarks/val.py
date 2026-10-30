from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from sslcon.tools import compute_stats
from kfolds.tools import compute_stats as compute_stats_kfolds

import yaml
import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

def categorize(freq):
    """
    Define threshold for different categories. Current option: 100, 1000.
    """
    if freq < 100:
        return "rare"
    elif freq < 1000:
        return "medium"
    else:
        return "common"

def analyze_frequencies(path):
    """
    Function, that creates plot of metric for classes of different frequencies
    Args:
        path (str): Path to folder with val_results.csv
    """
    df = pd.read_csv(f"{path}/val_results.csv")

    # Filter out the average row
    overall_row = df[df['class_id'] == -1].iloc[0]
    overall_map50_95 = overall_row['mAP50-95']
    df = df[(df['class_id'] != -1) & (df['nr_of_instances'] > 0) & (df['mAP50-95'] > -1)]

    # Plotting the classes per frequency category
    df['frequency_category'] = df['nr_of_instances'].apply(categorize)

    plt.figure(figsize=(12, 7))
    scatter = plt.scatter(
        df['nr_of_instances'], df['mAP50-95'],
        c=df['frequency_category'].map({'rare': 'red', 'medium': 'orange', 'common': 'green'}),
        label=None
    )

    # Annotate each point with class name
    for _, row in df.iterrows():
        plt.text(row['nr_of_instances'], row['mAP50-95'], row['class_name'], fontsize=8)

    # Vertical lines for frequency thresholds
    plt.axvline(x=100, color='gray', linestyle='--', label='Rare / Medium Threshold')
    plt.axvline(x=1000, color='gray', linestyle='--', label='Medium / Common Threshold')

    # Horizontal line for overall mAP@0.5-0.95 for all classes
    plt.axhline(y=overall_map50_95, color='blue', linestyle='--', label=f"Overall mAP@0.5 = {overall_map50_95:.3f}")

    plt.xscale('log')
    plt.xlabel('Number of Instances (log scale)')
    plt.ylabel('mAP@0.5-0.95')
    plt.title('YOLO Class Performance vs Frequency')
    plt.grid(True)


    # Report average mAP50-95 by frequency category
    avg_map_by_cat = df.groupby('frequency_category')['mAP50-95'].mean().sort_index()

    # Fit linear regression on log(instance count)
    X = np.log10(df['nr_of_instances'].values).reshape(-1, 1)
    y = df['mAP50-95'].values
    reg = LinearRegression().fit(X, y)

    slope = reg.coef_[0]
    intercept = reg.intercept_
    r2 = r2_score(y, reg.predict(X))

    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred = reg.predict(x_range.reshape(-1, 1))
    x_vals_log = 10 ** x_range  # Convert back to original scale for plotting

    print("\n Average mAP@0.5-0.95 by class frequency:")
    for category, avg_map in avg_map_by_cat.items():
        print(f"  {category.capitalize():<7}: {avg_map:.3f}")

    plt.plot(x_vals_log, y_pred, color='black', linestyle='-', linewidth=2, label='Linear Regression (log scale)')

    # Creating the legend
    equation = f"mAP@0.5-0.95 = {slope:.3f} × log₁₀(instances) + {intercept:.3f}\nR² = {r2:.3f}"
    legend_text = "\n".join([
        f"{cat.capitalize()}: mAP@0.5-0.95={avg_map_by_cat[cat]:.3f}"
        for cat in ['rare', 'medium', 'common'] if cat in avg_map_by_cat
    ])
    legend_text += f"\n{equation}"
    plt.legend([legend_text], loc='lower right')

    plt.tight_layout()
    plt.savefig(f"{path}/performance_per_frequency.png")



def validate_yolo(output: str, data: str, imgsz: int = 1920, device: str = 'cuda', kfolds_val: bool = False):
    """
    Benchmark a YOLOv8 model on a dataset.

    Args:
        model (str): Path to the YOLOv8 model file.
        data (str): Path to the dataset configuration file.
        imgsz (int): Image size for the benchmark. Default is 1920.
        device (str): Device to run the benchmark on. Default is 'cuda'.
        kfolds_val (bool): Signify that validation is run on Fold of kfolds. Default is False.
    """
    # Load a model
    model = YOLO(f"{output}/weights/best.pt")  # load an official model

    with open(data, 'r') as f:
        data_config = yaml.safe_load(f)
    data_path = data_config['path']

    if not os.path.isdir(f"{output}/VAL"):
        results = model.val(data=data, imgsz=imgsz, device=device, project=output, split="test", name="VAL")

        # Different validation for validation in k-Fold CV and normal GERALD training needed
        if not kfolds_val:
            compute_stats(results, data_path, f"{output}/VAL")
        else:
            compute_stats_kfolds(results, data, f"{output}/VAL")
        analyze_frequencies(f"{output}/VAL")
    else:
        print("VAL folder exists!")
