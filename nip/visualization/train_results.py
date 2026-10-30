import pandas as pd
import matplotlib.pyplot as plt
import os

# Set options for the plots
# Font size
SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 28

# Metric to plot and where the folders are
runs_dir = 'sslcon/runs'
metric = 'metrics/mAP50-95(B)'

def plot_default():
    """
    Plot the results.csv training metric over 200 epochs for all files in runs_dir.
    """
    plt.figure(figsize=(20, 12))

    # Iterate through each run folder to plot
    for run_name in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, run_name, 'GERALD/results.csv')
        if os.path.isfile(run_path):
            df = pd.read_csv(run_path)
            plt.plot(df[metric], label=run_name)

    plt.xticks(fontsize=SMALL_SIZE)
    plt.yticks(fontsize=SMALL_SIZE)
    plt.xlabel('Epoch', fontsize=SMALL_SIZE)
    plt.ylabel(metric, fontsize=SMALL_SIZE)
    plt.title(f'{metric} over Epochs', fontsize=MEDIUM_SIZE)
    plt.legend(loc="lower right", prop={'size': BIGGER_SIZE})
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("out/train_results.png")

def plot_early_stop():
    """
    Plot the results.csv training metric over 200 epochs, but with early stop.
    """
    plt.figure(figsize=(20, 12))

    for run_name in sorted(os.listdir(runs_dir)):
        run_path = os.path.join(runs_dir, run_name, 'GERALD/results.csv')
        if not os.path.isfile(run_path):
            print(run_path)
            continue

        df = pd.read_csv(run_path)

        # Find epoch of best performance and slice including the best
        best_epoch = df[metric].idxmax()
        best_val = df.loc[best_epoch, metric]
        df_trim = df.iloc[:best_epoch + 1]

        plt.plot(df_trim[metric], label=run_name)

        # Annotate best point
        plt.scatter(best_epoch, best_val, marker='o')
        plt.text(
            best_epoch, best_val,
            f"{best_val:.3f}",
            fontsize=SMALL_SIZE,
            va='bottom', ha='right'
        )

    plt.xticks(fontsize=SMALL_SIZE)
    plt.yticks(fontsize=SMALL_SIZE)
    plt.xlabel('Epoch', fontsize=SMALL_SIZE)
    plt.ylabel(metric, fontsize=SMALL_SIZE)
    plt.title(f'{metric} over Epochs', fontsize=MEDIUM_SIZE)
    plt.legend(loc="lower right", prop={'size': BIGGER_SIZE})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("out/train_results_early.png")

if __name__ == "__main__":
    """
    A quick script to generate the validation performance of the different methods over 200 epochs.
    """
    plot_default()
    plot_early_stop()
