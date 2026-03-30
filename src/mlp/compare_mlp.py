"""
AMINA FATMA KHAN
B00868087
March 27, 2026

- Compare Centralized MLP vs FedAvg MLP.
- Run AFTER both evaluate scripts (evaluate_centralized_mlp.py and evaluate_fedAvg.py)

Usage:
    python3 compare_mlp.py -cls mc
    python3 compare_mlp.py -cls b
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

os.makedirs('charts/compare_mlp', exist_ok=True)

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
    mlp_csv = f'charts/mlp/mlp_{suffix}_metrics.csv'
    fed_csv = f'charts/fedAvg/fedavg_{suffix}_metrics.csv'

    for path in [mlp_csv, fed_csv]:
        if not os.path.exists(path):
            print(f'ERROR: {path} not found. Run the evaluate scripts first.')
            sys.exit(1)

    mlp = pd.read_csv(mlp_csv).iloc[0]
    fed = pd.read_csv(fed_csv).iloc[0]

    """
    Load saved histories for training curves and time comparison
    """
    mlp_hist_path = f'saved_models/mlp_{suffix}_history.json'
    fed_hist_path = f'saved_models/fedavg_{suffix}_history.json'

    for path in [mlp_hist_path, fed_hist_path]:
        if not os.path.exists(path):
            print(f'ERROR: {path} not found. Run the main training scripts first.')
            sys.exit(1)

    with open(mlp_hist_path, 'r') as f:
        mlp_hist = json.load(f)
    with open(fed_hist_path, 'r') as f:
        fed_hist = json.load(f)

    mlp_time = mlp_hist.get('training_time', None)
    fed_time = fed_hist.get('training_time', None)

    print(f'Centralized MLP - Accuracy: {mlp["accuracy"]:.4f}  Macro-F1: {mlp["macro_f1"]:.4f}  Time: {mlp_time}s')
    print(f'FedAvg MLP      - Accuracy: {fed["accuracy"]:.4f}  Macro-F1: {fed["macro_f1"]:.4f}  Time: {fed_time}s')

    """
    1. Training curves
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1.1 MLP accuracy over epochs 
    axes[0].plot(mlp_hist['accuracy'],     label='Train')
    axes[0].plot(mlp_hist['val_accuracy'], label='Validation', linestyle='--')
    axes[0].set_title(f'Centralized MLP ({suffix.capitalize()})- Accuracy / Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 1.2 FedAvg accuracy over rounds
    rounds = list(range(1, len(fed_hist['round_accs']) + 1))
    axes[1].plot(rounds, fed_hist['round_accs'],   marker='o', label='Global Test')
    axes[1].plot(rounds, fed_hist['avg_val_accs'], marker='s', linestyle='--', label='Avg Client Val')
    axes[1].set_title(f'FedAvg MLP ({suffix.capitalize()})- Accuracy / Rounds')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(rounds)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 1.3. Training time between MLP and FedAvg
    if mlp_time is not None and fed_time is not None:
        bars = axes[2].bar(['Centralized MLP', 'FedAvg MLP'],
                           [mlp_time, fed_time],
                           color=['steelblue', 'darkorange'], width=0.4)
        for bar in bars:
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{bar.get_height()}s', ha='center', fontsize=11)
        axes[2].set_ylabel('Seconds')
        axes[2].set_title('Training Time Comparison')
        axes[2].grid(axis='y', alpha=0.3)

    fig.suptitle(f'Centralized MLP vs FedAvg ({suffix.capitalize()}- Training Curves & Time', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'charts/compare_mlp/comparison_mlp_vs_fedavg_{suffix}_training_curves_and_time.png')
    plt.show()

    """
    2. Overall metrics comparison from csv (accuracy, precision, recall, F1, FPR, FNR, AUC if binary)
    """
    metric_keys   = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'macro_fpr', 'macro_fnr']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'FPR', 'FNR']

    if cls == 'b' and 'auc' in mlp and 'auc' in fed:
        metric_keys.append('auc')
        metric_labels.append('AUC')

    mlp_vals = [mlp[k] for k in metric_keys]
    fed_vals = [fed[k] for k in metric_keys]

    x = np.arange(len(metric_labels))
    plt.figure(figsize=(11, 5))
    bars1 = plt.bar(x - 0.2, mlp_vals, 0.4, label='Centralized MLP')
    bars2 = plt.bar(x + 0.2, fed_vals, 0.4, label='FedAvg MLP')

    for bar in bars1 + bars2:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

    plt.xticks(x, metric_labels)
    plt.ylim(0, 1.2)
    plt.title(f'Centralized MLP vs FedAvg ({suffix.capitalize()})- Overall Metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'charts/compare_mlp/comparison_mlp_vs_fedavg_{suffix}_overall.png')
    plt.show()

    """
    3. Confusion matrices (from saved PNGs)
    """
    mlp_cm_path = f'charts/mlp/mlp_{suffix}_confusion_matrix.png'
    fed_cm_path = f'charts/fedAvg/fedavg_{suffix}_confusion_matrix.png'

    if os.path.exists(mlp_cm_path) and os.path.exists(fed_cm_path):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(Image.open(mlp_cm_path))
        axes[0].axis('off')
        axes[0].set_title('Centralized MLP')
        axes[1].imshow(Image.open(fed_cm_path))
        axes[1].axis('off')
        axes[1].set_title('FedAvg MLP')
        fig.suptitle(f'Centralized MLP vs FedAvg ({suffix.capitalize()})- Confusion Matrices')
        plt.tight_layout()
        plt.savefig(f'charts/compare_mlp/comparison_mlp_vs_fedavg_{suffix}_confusion.png')
        plt.show()
    else:
        print('Confusion matrix PNGs not found — skipping.')

    """
    4. Per-class F1 comparison
    """
    f1_cols = [c for c in mlp.index if c.startswith('f1_')]
    if f1_cols:
        class_labels = [c.replace('f1_', '') for c in f1_cols]
        mlp_f1 = [mlp[c] for c in f1_cols]
        fed_f1 = [fed[c] for c in f1_cols]

        x = np.arange(len(class_labels))
        plt.figure(figsize=(9, 5))
        plt.bar(x - 0.2, mlp_f1, 0.4, label='Centralized MLP')
        plt.bar(x + 0.2, fed_f1, 0.4, label='FedAvg MLP')
        plt.xticks(x, class_labels, rotation=15)
        plt.ylim(0, 1.15)
        plt.title(f'Centralized MLP vs FedAvg ({suffix.capitalize()})- Per-Class F1')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'charts/compare_mlp/comparison_mlp_vs_fedavg_{suffix}_per_class_f1.png')
        plt.show()

    """
    5. ROC curves (binary only)
    """
    if cls == 'b':
        mlp_roc_path = f'charts/mlp/mlp_{suffix}_roc_curve.png'
        fed_roc_path = f'charts/fedAvg/fedavg_{suffix}_roc_curve.png'

        if os.path.exists(mlp_roc_path) and os.path.exists(fed_roc_path):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(Image.open(mlp_roc_path))
            axes[0].axis('off')
            axes[0].set_title('Centralized MLP')
            axes[1].imshow(Image.open(fed_roc_path))
            axes[1].axis('off')
            axes[1].set_title('FedAvg MLP')
            fig.suptitle(f'ROC Curve Comparison ({suffix.capitalize()})- MLP vs FedAvg')
            plt.tight_layout()
            plt.savefig(f'charts/compare_mlp/comparison_mlp_vs_fedavg_{suffix}_roc.png')
            plt.show()
        else:
            print('ROC curve PNGs not found — skipping.')

    print('\nDone! Comparison charts saved.')