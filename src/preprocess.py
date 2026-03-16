"""
Ava Powelson
B00802243
March 12, 2026

See README.md for references.
"""
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

"""
Load WSN-DS from csv
"""
def load_data(file='../data/wsn-ds.csv'):
    df = pd.read_csv(file)
    print(f"Number of Rows x Columns: {df.shape[0]} x {df.shape[1]}\n")
    print(f"Class distribution:\n {df.iloc[:, -1].value_counts()}\n")

    # No null cells encountered
    # print(df.isnull().sum())

    return df

"""
Balance using SMOTE
"""
def balance_data(df):
    data = df.iloc[:, :-1]
    lbl = df.iloc[:, -1]
    smote = SMOTE(random_state=42)
    data_resampled, lbl_resampled = smote.fit_resample(data, lbl)
    df_resampled = pd.concat([data_resampled, lbl_resampled], axis=1)
    print(f"Balanced data shape: {df_resampled.shape}")
    print(f"Balanced class distribution:\n {df_resampled.iloc[:, -1].value_counts()}\n")
    
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.countplot(x=lbl_resampled)
    # plt.title("Class Distro after SMOTE")
    # plt.show()

    """
    Handle Outliers
    - IQR 25-75 removes everything except TDMA and Normal, so we'll just keep everything
    """
    # numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # q1 = df[numerical_cols].quantile(0.25)
    # q3 = df[numerical_cols].quantile(0.75)
    # iqr = q3 - q1
    # df_no_outliers = df[~((df[numerical_cols] < (q1 - 1.5*iqr)) | (df[numerical_cols] > (q3 + 1.5*iqr))).any(axis=1)]
    # print(f"Outliers Removed Rows: {df.shape[0]}")

    return df_resampled

def split_data(df, test_size=0.3, target_column="label"):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(X, y, test_size=test_size)
