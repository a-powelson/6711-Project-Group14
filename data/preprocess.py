"""
Ava Powelson
B00802243
March 12, 2026

Following the steps included in the WSN-DS repository.
https://github.com/m-zeeshan555/WSN-DS/blob/main/Wireless%20Sensor%20Network%20Project%20Notebook.pdf
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

df = pd.read_csv("wsn-ds.csv")

print(f"Number of Rows x Columns: {df.shape[0]} x {df.shape[1]}\n")
print(f"Class distribution:\n {df.iloc[:, -1].value_counts()}\n")

# No null cells encountered
# print(df.isnull().sum())

# Remove outliers
# numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
# q1 = df[numerical_cols].quantile(0.25)
# q3 = df[numerical_cols].quantile(0.75)
# iqr = q3 - q1
# df_no_outliers = df[~((df[numerical_cols] < (q1 - 1.5*iqr)) | (df[numerical_cols] > (q3 + 1.5*iqr))).any(axis=1)]

# print(f"Outliers Removed Rows: {df.shape[0]}")

# Class distribution
class_counts = df['label'].value_counts()
print(f"Class distro:\n {class_counts}")

# Balance using SMOTE
data = df.iloc[:, :-1]
lbl = df.iloc[:, -1]

smote = SMOTE(random_state=42)
data_resampled, lbl_resampled = smote.fit_resample(data, lbl)

# sns.countplot(x=lbl_resampled)
# plt.title("Class Distro after SMOTE")
# plt.show()

df_resampled = pd.concat([data_resampled, lbl_resampled], axis=1)
print(f"Resampled data shape: {df_resampled.shape}")

# Feature Engineering TBD
...