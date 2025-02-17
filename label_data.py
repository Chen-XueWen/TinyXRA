import os
from tqdm import tqdm
import numpy as np
import pandas as pd

years = range(2024, 2018, -1)
# Ensure the directory exists
if not os.path.isdir('datasets/labelled/'):
    os.makedirs('datasets/labelled/', exist_ok=True)

# Use tqdm correctly
for year in tqdm(years, desc="Processing Years", unit="year"):
    print(f"Processing Year {year}")
    df = pd.read_json(f"datasets/10k_volatility/{year}.json")
    # For nan sortino value, it means the stock does not have negative returns, therefore it is infinite (replace with some big numbers)
    df["Sortino_value"] = df["Sortino_value"].fillna(100)
    # 0-30%, 30-70%, 70-100% labels=[0, 1, 2]  # Labels for the bins
    df['Std_label'] = pd.qcut(df['Std_value'], q=[0, 0.3, 0.7, 1], labels=[0, 1, 2])
    df['Skewness_label'] = pd.qcut(df['Skewness_value'], q=[0, 0.3, 0.7, 1], labels=[0, 1, 2])
    df['Kurtosis_label'] = pd.qcut(df['Kurtosis_value'], q=[0, 0.3, 0.7, 1], labels=[0, 1, 2])
    df['Sortino_label'] = pd.qcut(df['Sortino_value'], q=[0, 0.3, 0.7, 1], labels=[0, 1, 2])
    df = df.drop(columns=['Std_value', 'Skewness_value', 'Kurtosis_value', 'Sortino_value'])
    df.to_json(f"./datasets/labelled/{year}.json", 
           orient="records", 
           indent=4,
           double_precision=15,
           force_ascii=False)
    
print("Processing complete!")