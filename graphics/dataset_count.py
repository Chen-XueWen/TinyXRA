import os
from tqdm import tqdm
import numpy as np
import pandas as pd

percentiles = [0.3, 0.5, 0.7]  # Define percentiles
years = range(2024, 2012, -1)

for year in tqdm(years, desc="Processing Years", unit="year"):
    print(f"Processing Year {year}")
    df = pd.read_json(f"datasets/10k_volatility/{year}.json")

   # Drop NaN values properly
    df = df.dropna(subset=["Skewness_value", "Kurtosis_value"])

    # For NaN Sortino values, replace them with 100 (infinite assumption)
    df["Sortino_value"] = df["Sortino_value"].fillna(100)

    # 0-30%, 30-70%, 70-100% labels=[0, 1, 2]  # Labels for the bins
    df['Std_label'] = pd.qcut(df['Std_value'], q=[0, 0.3, 0.7, 1], labels=[0, 1, 2])
    df['Skewness_label'] = pd.qcut(df['Skewness_value'], q=[0, 0.3, 0.7, 1], labels=[0, 1, 2])
    df['Kurtosis_label'] = pd.qcut(df['Kurtosis_value'], q=[0, 0.3, 0.7, 1], labels=[0, 1, 2])
    df['Sortino_label'] = pd.qcut(df['Sortino_value'], q=[0, 0.3, 0.7, 1], labels=[0, 1, 2])

    # Compute percentiles for relevant columns
    std_percentiles = df["Std_value"].quantile(percentiles)
    skewness_percentiles = df["Skewness_value"].quantile(percentiles)
    kurtosis_percentiles = df["Kurtosis_value"].quantile(percentiles)
    sortino_percentiles = df["Sortino_value"].quantile(percentiles)

    std_counts = df['Std_label'].value_counts().sort_index()

    # Print or store results
    print(f"Year {year}:")
    print(f"Number of rows:\n{df.shape[0]}")
    print(f"Std_value Bins:\n{std_counts}")
    print(f"Std_value Percentiles:\n{std_percentiles}")
    print(f"Skewness_value Percentiles:\n{skewness_percentiles}")
    print(f"Kurtosis_value Percentiles:\n{kurtosis_percentiles}")
    print(f"Sortino_value Percentiles:\n{sortino_percentiles}")
    
print("Processing complete!")