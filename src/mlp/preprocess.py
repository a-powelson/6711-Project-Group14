"""
Ava Powelson
B00802243
March 12, 2026

See README.md for references.

Tuneable parameters:
- normalization method
- test size
- balancing method
"""
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

"""
Load WSN-DS from csv
"""
def load_data(file='../data/wsn-ds.csv'):
    df = pd.read_csv(file)
    print(f"Number of Rows x Columns: {df.shape[0]} x {df.shape[1]}\n")
    print(f"Class distribution:\n {df.iloc[:, -1].value_counts()}\n")

    # cleanup column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )

    # No null cells encountered
    # print(df.isnull().sum())

    # Convert labels to ints
    le = LabelEncoder()
    le.fit(df.label)
    df['label'] = le.transform(df.label)

    return df

"""
Normalize
"""
def normalize_data(x):
    scaler = MinMaxScaler()
    return scaler.fit_transform(x)

"""
Balance using SMOTE
"""
def balance_data(x, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(x, y)

"""
Split into train and test groups
"""
def split_data(df, test_size=0.3, target_column="label"):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(X, y, test_size=test_size)
