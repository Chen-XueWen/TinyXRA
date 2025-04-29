import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 16,     # Title font size
    'axes.labelsize': 16,     # X/Y label font size
    'xtick.labelsize': 16,    # X-axis tick label font size
    'ytick.labelsize': 16,    # Y-axis tick label font size
    'legend.fontsize': 16     # Legend font size
})

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate risk metric plots.')
parser.add_argument("--risk_metric", choices=["std", "skew", "kurt", "sortino"], type=str, required=True)
args = parser.parse_args()
risk_metrics = args.risk_metric

seeds = [98, 83, 62, 42, 21]
years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
thresholds = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']

# Define the metrics and their positions in the data[key] list
metrics_info = {
    "f1": 1,
    "spearman": 2,
    "kendall": 3
}

# Make sure output directory exists
os.makedirs("images", exist_ok=True)

for metric_name, metric_idx in metrics_info.items():
    plt.figure(figsize=(10, 6))

    for year in years:
        metric_dict = {t: [] for t in thresholds}

        for seed in seeds:
            explain_result_loc = f"explain_experiments/words/{risk_metrics}/Y{year}S{seed}.json"
            with open(explain_result_loc, 'r') as file:
                data = json.load(file)

            for key in data:
                metric_dict[key].append(data[key][metric_idx])

        x = [float(k) * 100 for k in metric_dict.keys()]
        y = [np.mean(v) for v in metric_dict.values()]
        yerr = [np.std(v) for v in metric_dict.values()]

        plt.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, label=f'Year {year}')
        #plt.plot(x, y, marker='o', label=f'Year {year}')
        #plt.fill_between(x, np.array(y) - np.array(yerr), np.array(y) + np.array(yerr), alpha=0.2)

    plt.xlabel('% of top attended words removed')
    plt.ylabel(f'{metric_name.capitalize()} Score')
    #plt.title(f'{metric_name.capitalize()} Score vs Removed Words ({risk_metrics})')
    plt.legend(loc='lower left')
    plt.grid(True)

    # Save each metric's figure
    plt.savefig(f'images/{risk_metrics}_{metric_name}_multi_year_words_plot.png', dpi=300, bbox_inches='tight')