"""normalize_data.py
Reads LifestyleDAG.csv, z-scores every numeric column except 'precipitationFlag',
then writes LifestyleDAG_norm.csv.  Run this after generate_synthetic.py.
"""
import pandas as pd
import numpy as np

RAW_FILE = "LifestyleDAG.csv"
NORM_FILE = "LifestyleDAG_norm.csv"


def main():
    df = pd.read_csv(RAW_FILE)

    # Identify numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Binary flag(s) we leave untouched (avoid turning 0/1 into z-scores)
    binary_cols = [
        col for col in num_cols
        if set(df[col].unique()).issubset({0, 1}) and col == "precipitationFlag"
    ]

    cols_to_scale = [c for c in num_cols if c not in binary_cols]

    df_scaled = df.copy()
    for col in cols_to_scale:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            # constant column, set to 0
            df_scaled[col] = 0.0
        else:
            df_scaled[col] = (df[col] - mean) / std

    df_scaled.to_csv(NORM_FILE, index=False)
    print(f"Saved normalized file to {NORM_FILE} with {len(df_scaled)} rows")


if __name__ == "__main__":
    main() 