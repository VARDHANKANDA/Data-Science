"""
================================================================================
WEEK 1 DATA SCIENCE APPRENTICESHIP: DATA ACQUISITION, CLEANING & EDA PIPELINE
================================================================================
Dataset: UCI Census Income (Adult) Benchmark Dataset
Author: Lead Data Science Apprentice
Environment: Python 3.12, Pandas 2.2, NumPy 2.0, Matplotlib 3.10, Seaborn 0.13
================================================================================
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("="*70)
    print("WEEK 1: DATA ACQUISITION, CLEANING & EXPLORATORY DATA ANALYSIS PIPELINE")
    print("="*70)
    
    # ---------------------------------------------------------
    # 1. DATA ACQUISITION
    # ---------------------------------------------------------
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    local_path = "adult.data"
    
    if not os.path.exists(local_path):
        print(f"Downloading raw dataset from {data_url}...")
        urllib.request.urlretrieve(data_url, local_path)
        print("Download completed successfully.")
    else:
        print(f"Dataset already present locally at '{local_path}'.")
        
    col_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    
    df_raw = pd.read_csv(local_path, header=None, names=col_names, skipinitialspace=True)
    print(f"Raw Dataset Ingested: {df_raw.shape[0]:,} rows by {df_raw.shape[1]} columns.")
    
    # ---------------------------------------------------------
    # 2. DATA QUALITY AUDIT & CLEANING
    # ---------------------------------------------------------
    df_clean = df_raw.copy()
    
    # 2.1 Strip Whitespace
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        
    # 2.2 Handle Missingness ('?' markers)
    for col in ['workclass', 'occupation', 'native_country']:
        df_clean[col] = df_clean[col].replace('?', 'Unknown')
        
    # 2.3 Deduplication
    init_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    print(f"Removed {init_rows - len(df_clean)} duplicate rows. Cleaned row count: {len(df_clean):,}")
    
    # 2.4 Binary Target Flag
    df_clean['income_binary'] = (df_clean['income'] == '>50K').astype(int)
    
    # ---------------------------------------------------------
    # 3. SUMMARY STATISTICS
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("DESCRIPTIVE STATISTICS (NUMERICAL FEATURES)")
    print("="*50)
    print(df_clean.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])
    
    print("\n" + "="*50)
    print("TARGET CLASS DISTRIBUTION")
    print("="*50)
    print(df_clean['income'].value_counts(normalize=True) * 100)
    
    print("\nPipeline executed cleanly.")

if __name__ == "__main__":
    main()
