"""
AMINA FATMA KHAN
B00868087
March 27, 2026

- Evaluate the FedAvg MLP model and generate charts.
- Run main_fedAvg.py first.

Usage:
    python3 evaluate_fedAvg.py -cls mc
    python3 evaluate_fedAvg.py -cls b
"""

from src.preprocessing.preprocess import *
from src.preprocessing.args import *

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

os.makedirs('charts/fedAvg', exist_ok=True)

CLASS_NAMES_MC = ["Blackhole", "Flooding", "Grayhole", "Normal", "Scheduling"]
CLASS_NAMES_B  = ["Normal", "Attack"]

if __name__ == '__main__':
    args   = args_parser()
    cls    = args.cls
    names  = CLASS_NAMES_MC if cls == 'mc' else CLASS_NAMES_B
    suffix = 'multiclass' if cls == 'mc' else 'binary'

    """
    Load saved model 
    """
    model_path = f'saved_models/fedavg_{suffix}_model.keras'
    if not os.path.exists(model_path):
        print(f'ERROR: {model_path} not found. Run main_fedAvg.py -cls {cls} first.')
        sys.exit(1)
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")

    """
    Load saved history
    """
    history_path = f'saved_models/fedavg_{suffix}_history.json'
    if not os.path.exists(history_path):
        print(f'ERROR: {history_path} not found. Run main_fedAvg.py -cls {cls} first.')
        sys.exit(1)
    with open(history_path, 'r') as f:
        history = json.load(f)
    training_time = history.get('training_time', None)
    print(f"Loaded history from {history_path}")

    """
    Data
    """
    gbl_data = load_data('data/wsn-ds.csv', cls=cls)
    x_train_gbl, x_test_gbl, y_train_gbl, y_test_gbl = split_data(gbl_data, test_size=0.3)
    x_test_gbl = normalize_data(x_test_gbl)

    """
    Predict and compute metrics
    """
    y_pred_probs = model.predict(x_test_gbl)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    report       = classification_report(y_test_gbl, y_pred, output_dict=True)
    cm           = confusion_matrix(y_test_gbl, y_pred)

    """
    Compute FPR/FNR per class from confusion matrix
    """
    fpr_per_class = []
    fnr_per_class = []
    total = cm.sum()
    for i in range(len(names)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        fpr_per_class.append(round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4))
        fnr_per_class.append(round(fn / (fn + tp) if (fn + tp) > 0 else 0, 4))

    # 1. Training curves
    rounds = list(range(1, len(history['round_accs']) + 1))
    plt.figure(figsize=(12, 4))

    # 1.1 Accuracy over epochs
    plt.subplot(1, 2, 1)
    plt.plot(rounds, history['round_accs'],   marker='o', label='Global Test Accuracy')
    plt.plot(rounds, history['avg_val_accs'], marker='s', linestyle='--', label='Avg Client Val Accuracy')
    plt.title('Accuracy per Round')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.xticks(rounds)
    plt.legend()

    # 1.2 Loss over epochs
    plt.subplot(1, 2, 2)
    plt.plot(rounds, history['round_losses'],   marker='o', color='tab:red',    label='Global Test Loss')
    plt.plot(rounds, history['avg_val_losses'], marker='s', color='tab:orange', linestyle='--', label='Avg Client Val Loss')
    plt.title('Loss per Round')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.xticks(rounds)
    plt.legend()

    plt.suptitle(f'FedAvg MLP {suffix.capitalize()} - Training Curves')
    plt.tight_layout()
    plt.savefig(f'charts/fedAvg/fedavg_{suffix}_training_curves.png')
    plt.show()

    """
    2. Confusion matrix
    """
    plt.figure(figsize=(max(6, len(names) * 1.4), max(5, len(names) * 1.2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=names, yticklabels=names)
    plt.title(f'FedAvg MLP {suffix.capitalize()} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'charts/fedAvg/fedavg_{suffix}_confusion_matrix.png')
    plt.show()

    """
    3. Per-class precision, recall, F1, FPR, FNR
    """
    precision = [report[str(i)]['precision'] for i in range(len(names))]
    recall    = [report[str(i)]['recall']    for i in range(len(names))]
    f1        = [report[str(i)]['f1-score']  for i in range(len(names))]

    x     = np.arange(len(names))
    width = 0.15

    plt.figure(figsize=(max(9, len(names) * 2), 5))
    plt.bar(x - 2*width, precision,     width, label='Precision')
    plt.bar(x - width,   recall,        width, label='Recall')
    plt.bar(x,           f1,            width, label='F1')
    plt.bar(x + width,   fpr_per_class, width, label='FPR', color='tomato')
    plt.bar(x + 2*width, fnr_per_class, width, label='FNR', color='darkorange')
    plt.xticks(x, names, rotation=15)
    plt.ylim(0, 1.15)
    plt.title(f'FedAvg MLP {suffix.capitalize()} - Per-Class Metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'charts/fedAvg/fedavg_{suffix}_per_class_metrics.png')
    plt.show()


    """
    4. ROC curve (binary only) 
    """
    roc_auc = None
    if cls == 'b':
        fpr_curve, tpr_curve, _ = roc_curve(y_test_gbl, y_pred_probs[:, 1])
        roc_auc = auc(fpr_curve, tpr_curve)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr_curve, tpr_curve, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.title('FedAvg MLP Binary - ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig('charts/fedAvg/fedavg_binary_roc_curve.png')
        plt.show()

    """
    Save metrics CSV and classification report JSON for comparison with FedAvg
    """
    final_loss, final_acc = model.evaluate(x_test_gbl, y_test_gbl, verbose=0)

    metrics = {
        'model':           'FedAvg MLP',
        'cls':             cls,
        'accuracy':        round(final_acc, 4),
        'macro_precision': round(report['macro avg']['precision'], 4),
        'macro_recall':    round(report['macro avg']['recall'],    4),
        'macro_f1':        round(report['macro avg']['f1-score'],  4),
        'macro_fpr':       round(np.mean(fpr_per_class), 4),
        'macro_fnr':       round(np.mean(fnr_per_class), 4),
        'training_time':   training_time,
    }
    if roc_auc is not None:
        metrics['auc'] = round(roc_auc, 4)

    for i, name in enumerate(names):
        metrics[f'f1_{name}']  = round(report[str(i)]['f1-score'], 4)
        metrics[f'fpr_{name}'] = fpr_per_class[i]
        metrics[f'fnr_{name}'] = fnr_per_class[i]

    pd.DataFrame([metrics]).to_csv(f'charts/fedAvg/fedavg_{suffix}_metrics.csv', index=False)
    with open(f'charts/fedAvg/fedavg_{suffix}_report.json', 'w') as f:
        json.dump(report, f, indent=4)

    print(f'\nAll charts and metrics saved to charts/fedAvg/')
    print('Done!')