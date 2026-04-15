"""
AMINA FATMA KHAN
B00868087
March 27, 2026

- Dataset Analysis: Class Imbalance and Throughput (WSN-DS dataset)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('charts/dataset', exist_ok=True)

"""
Load raw data 
"""
raw = pd.read_csv('data/wsn-ds.csv')
raw.columns = raw.columns.str.strip().str.replace(' ', '_').str.replace('-', '_').str.replace('/', '_')

"""
Binary and multiclass label columns
"""
raw['binary_label'] = raw['label'].apply(lambda x: 'Normal' if x == 'Normal' else 'Attack')

"""
1. Class distribution — Binary
"""
binary_counts = raw['binary_label'].value_counts()

plt.figure(figsize=(6, 5))
bars = plt.bar(binary_counts.index, binary_counts.values,
               color=['tomato', 'steelblue'], width=0.4)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
             f'{int(bar.get_height()):,}', ha='center', fontsize=10)
plt.title('Dataset Class Distribution - Binary')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.tight_layout()
plt.savefig('charts/dataset/dataset_class_distribution_binary.png')
plt.show()
print(f"Binary class distribution:\n{binary_counts}\n")

"""
2. Class distribution — Multiclass
"""
multi_counts = raw['label'].value_counts()

plt.figure(figsize=(8, 5))
bars = plt.bar(multi_counts.index, multi_counts.values, color='steelblue', width=0.5)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
             f'{int(bar.get_height()):,}', ha='center', fontsize=9)
plt.title('Dataset Class Distribution - Multiclass')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('charts/dataset/dataset_class_distribution_multiclass.png')
plt.show()
print(f"Multiclass class distribution:\n{multi_counts}\n")

"""
3. Throughput — Binary (Attack vs Normal)
"""
sent_b     = raw.groupby('binary_label')['DATA_S'].mean()
received_b = raw.groupby('binary_label')['DATA_R'].mean()

x     = np.arange(len(sent_b))
width = 0.35

plt.figure(figsize=(7, 5))
plt.bar(x - width/2, sent_b.values,     width, label='Avg DATA_S (Sent)',     color='steelblue')
plt.bar(x + width/2, received_b.values, width, label='Avg DATA_R (Received)', color='seagreen')
plt.xticks(x, sent_b.index, rotation=0)
plt.ylabel('Average Packets')
plt.title('Dataset Throughput - Binary (Avg Packets Sent vs Received)')
plt.legend()
plt.tight_layout()
plt.savefig('charts/dataset/dataset_throughput_binary.png')
plt.show()

"""
4. Throughput — Multiclass (all classes)
"""
sent_mc     = raw.groupby('label')['DATA_S'].mean()
received_mc = raw.groupby('label')['DATA_R'].mean()

x = np.arange(len(sent_mc))

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, sent_mc.values,     width, label='Avg DATA_S (Sent)',     color='steelblue')
plt.bar(x + width/2, received_mc.values, width, label='Avg DATA_R (Received)', color='seagreen')
plt.xticks(x, sent_mc.index, rotation=15)
plt.ylabel('Average Packets')
plt.title('Dataset Throughput - Multiclass (Avg Packets Sent vs Received)')
plt.legend()
plt.tight_layout()
plt.savefig('charts/dataset/dataset_throughput_multiclass.png')
plt.show()

print('Done! All dataset charts saved to charts/dataset/')
