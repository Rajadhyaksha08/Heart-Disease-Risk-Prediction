import os
import pandas as pd
import numpy as np

COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
    'ca', 'thal', 'target'
]

def preprocess_data(raw_path: str = 'data/raw/processed.cleveland.data', 
                    output_path: str = 'data/processed/heart_processed.csv') -> pd.DataFrame:
    """
    Loads raw Cleveland heart disease dataset, cleans missing values,
    converts column types, standardizes target variable to binary classification,
    and saves the processed dataset.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")
        
    print("--- Preprocessing Cleveland Heart Disease Dataset ---")
    
    # 1. Load dataset with proper column names and missing value handling
    df = pd.read_csv(raw_path, header=None, names=COLUMNS, na_values='?')
    initial_shape = df.shape
    print(f"Loaded raw dataset with shape: {initial_shape}")
    
    # 2. Convert ca and thal to numeric data types
    df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
    df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
    
    # 3. Missing values summary and imputation (mode imputation for categorical/discrete attributes)
    missing_summary = df.isna().sum()
    total_missing = missing_summary.sum()
    print(f"Missing values detected: {total_missing}")
    for col, count in missing_summary[missing_summary > 0].items():
        print(f"  - Column '{col}': {count} missing value(s)")
        # Impute missing values with mode
        col_mode = df[col].mode()[0]
        df[col] = df[col].fillna(col_mode)
        print(f"    Imputed missing values in '{col}' using mode ({col_mode})")
        
    # 4. Remove duplicate rows if any
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicate_count} duplicate row(s).")
    else:
        print("No duplicate rows found.")
        
    # 5. Convert target variable to binary classification (0 = no disease, 1-4 = disease)
    df['target'] = (df['target'] > 0).astype(int)
    
    # 6. Ensure output directory exists and save processed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # 7. Print summary
    print("\n--- Cleaning Summary ---")
    print(f"Initial Shape        : {initial_shape}")
    print(f"Final Processed Shape: {df.shape}")
    print(f"Columns              : {list(df.columns)}")
    print(f"Data Types           :\n{df.dtypes.to_string()}")
    print("\nTarget Class Distribution:")
    print(df['target'].value_counts().rename({0: '0 (No Disease)', 1: '1 (Heart Disease)'}).to_string())
    print(f"\nSaved processed dataset to: {output_path}")
    print("------------------------\n")
    
    return df

if __name__ == '__main__':
    preprocess_data()

