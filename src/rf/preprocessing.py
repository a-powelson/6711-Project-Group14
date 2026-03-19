"""
Amina Fatma Khan
B00868087
March 18, 2026

See README.md for references.
"""

import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv("WSN-DS.csv")

    print(df.shape)
    print(df.head())
    print(df.columns.tolist())

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )
    print(df.columns.tolist())
  
    print(df.info())
    print(df.isnull().sum())
    print(df.duplicated().sum())
    print(df['label'].value_counts())

    # Remove duplicate rows, reset index and drop missing values
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna()

    # Binary labels - "Normal" or "Attack"
    df['binary_label'] = df['label'].apply(lambda x: 'Normal' if x == 'Normal' else 'Attack')
    print(df['binary_label'].value_counts())

    drop_cols = ['id', 'label', 'binary_label']
    X = df.drop(columns=drop_cols)
    y = df['binary_label']

    return df, X, y
