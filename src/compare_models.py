"""
AMINA FATMA KHAN
B00868087
- Compare Random Forest vs FedAvg MLP.
- Run AFTER both evaluate scripts (evaluate_rf.py and evaluate_fedAvg.py)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

os.makedirs('charts/compare_models', exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cls', type=str, default='mc', help='binary (b) or multiclass (mc)')
    return parser.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    cls    = args.cls
    suffix = 'multiclass' if cls == 'mc' else 'binary'

    """
    Load models' evaluation CSVs with overall metrics
    """
    rf_csv  = f'charts/rf/rf_{suffix}_metrics.csv'
    fed_csv = f'charts/fedAvg/fedavg_{suffix}_metrics.csv'

    for path in [rf_csv, fed_csv]:
        if not os.path.exists(path):
            print(f'ERROR: {path} not found. Run the evaluate scripts first.')
            sys.exit(1)

    rf  = pd.read_csv(rf_csv).iloc[0]
    fed = pd.read_csv(fed_csv).iloc[0]

    print(f'Random Forest - Accuracy: {rf["accuracy"]:.4f}  Macro-F1: {rf["macro_f1"]:.4f}')
    print(f'FedAvg MLP    - Accuracy: {fed["accuracy"]:.4f}  Macro-F1: {fed["macro_f1"]:.4f}')

    """
    1. Overall metrics comparison from csv (accuracy, precision, recall, F1, FPR, FNR, AUC if binary)
    """
    metric_keys   = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']

    if cls == 'b' and 'auc' in rf.index and 'auc' in fed.index:
        metric_keys.append('auc')
        metric_labels.append('AUC')

    rf_vals  = [rf[k]  for k in metric_keys]
    fed_vals = [fed[k] for k in metric_keys]

    x = np.arange(len(metric_labels))
    plt.figure(figsize=(9, 5))
    bars1 = plt.bar(x - 0.2, rf_vals,  0.4, label='Random Forest')
    bars2 = plt.bar(x + 0.2, fed_vals, 0.4, label='FedAvg MLP')

    for bar in bars1 + bars2:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    plt.xticks(x, metric_labels)
    plt.ylim(0, 1.2)
    plt.title(f'Random Forest vs FedAvg ({suffix.capitalize()})- Overall Metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'charts/compare_models/comparison_{suffix}_rf_vs_fedavg_overall.png')
    plt.show()

    """
    2. Confusion matrices (from saved PNGs)
    """
    rf_cm_path  = f'charts/rf/rf_{suffix}_confusion_matrix.png'
    fed_cm_path = f'charts/fedAvg/fedavg_{suffix}_confusion_matrix.png'

    if os.path.exists(rf_cm_path) and os.path.exists(fed_cm_path):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(Image.open(rf_cm_path))
        axes[0].axis('off')
        axes[0].set_title('Random Forest')
        axes[1].imshow(Image.open(fed_cm_path))
        axes[1].axis('off')
        axes[1].set_title('FedAvg MLP')
        fig.suptitle(f'Confusion Matrix Comparison ({suffix.capitalize()})- RF vs FedAvg')
        plt.tight_layout()
        plt.savefig(f'charts/compare_models/comparison_{suffix}_rf_vs_fedavg_confusion.png')
        plt.show()
    else:
        print('Confusion matrix PNGs not found — skipping that chart.')

    """
    3. Per-class F1 comparison
    """
    f1_cols = [c for c in rf.index if c.startswith('f1_')]
    if f1_cols:
        class_labels = [c.replace('f1_', '') for c in f1_cols]
        rf_f1  = [rf[c]  for c in f1_cols]
        fed_f1 = [fed[c] for c in f1_cols]

        x = np.arange(len(class_labels))
        plt.figure(figsize=(9, 5))
        plt.bar(x - 0.2, rf_f1,  0.4, label='Random Forest')
        plt.bar(x + 0.2, fed_f1, 0.4, label='FedAvg MLP')
        plt.xticks(x, class_labels, rotation=15)
        plt.ylim(0, 1.15)
        plt.title(f'Random Forest vs FedAvg ({suffix.capitalize()})- Per-Class F1')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'charts/compare_models/comparison_{suffix}_rf_vs_fedavg_per_class_f1.png')
        plt.show()

    """
    4. ROC curves (binary only)
    """
    if cls == 'b':
        rf_roc_path  = f'charts/rf/rf_{suffix}_roc_curve.png'
        fed_roc_path = f'charts/fedAvg/fedavg_{suffix}_roc_curve.png'

        if os.path.exists(rf_roc_path) and os.path.exists(fed_roc_path):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(Image.open(rf_roc_path))
            axes[0].axis('off')
            axes[0].set_title('Random Forest')
            axes[1].imshow(Image.open(fed_roc_path))
            axes[1].axis('off')
            axes[1].set_title('FedAvg MLP')
            fig.suptitle(f'ROC Curve Comparison ({suffix.capitalize()})- RF vs FedAvg')
            plt.tight_layout()
            plt.savefig(f'charts/compare_models/comparison_{suffix}_rf_vs_fedavg_roc.png')
            plt.show()
        else:
            print('ROC curve PNGs not found — skipping that chart.')

    print('\nDone! Comparison charts saved.')