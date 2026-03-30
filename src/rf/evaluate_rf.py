"""
AMINA FATMA KHAN
B00868087
March 27, 2026

- Evaluate the Random Forest model and generate charts.
- Run rf_modelp.py first.

Usage:
    python evaluate_rf.py -cls mc
    python evaluate_rf.py -cls b
"""

from src.preprocessing.preprocess import *
from src.preprocessing.args import *

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)

os.makedirs('charts/rf', exist_ok=True)

CLASS_NAMES_MC = ["Blackhole", "Flooding", "Grayhole", "Normal", "Scheduling"]
CLASS_NAMES_B  = ["Attack", "Normal"]  

if __name__ == '__main__':
    args   = args_parser()
    cls    = args.cls
    suffix = 'multiclass' if cls == 'mc' else 'binary'
    names  = CLASS_NAMES_MC if cls == 'mc' else CLASS_NAMES_B

    """
    Load saved model
    """
    model_path = f'saved_models/rf_{suffix}_model.pkl'
    if not os.path.exists(model_path):
        print(f'ERROR: {model_path} not found. Run rf_model.py -cls {cls} first.')
        sys.exit(1)
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    """
    Load history
    """
    with open(f'saved_models/rf_{suffix}_history.json', 'r') as f:
        history = json.load(f)
    training_time = history.get('training_time', None)

    """
    Data
    """
    data = load_data('data/wsn-ds.csv', cls=cls)
    X_train, X_test, y_train, y_test = split_data(data, test_size=0.3)
    feature_cols = X_test.columns.tolist()
    X_test = normalize_data(X_test)

    # Rename columns - for feature importance chart labels
    column_mapping = {
        'id':               'Node ID',
        'Time':             'Simulation Time',
        'Is_CH':            'Is Cluster Head (0/1)',
        'who_CH':           'Assigned Cluster Head',
        'Dist_To_CH':       'Distance to Cluster Head',
        'ADV_S':            'Cluster Head Ads Sent',
        'ADV_R':            'Cluster Head Ads Received',
        'JOIN_S':           'Join Requests Sent to CH',
        'JOIN_R':           'Join Requests Received by CH',
        'SCH_S':            'TDMA Schedule Messages Sent',
        'SCH_R':            'TDMA Schedule Messages Received',
        'Rank':             'TDMA Slot Rank',
        'DATA_S':           'Data Packets Sent to CH',
        'DATA_R':           'Data Packets Received from CH',
        'Data_Sent_To_BS':  'Data Packets Forwarded to Base Station',
        'dist_CH_To_BS':    'Distance from CH to Base Station',
        'send_code':        'Transmission Round Code',
        'Consumed_Energy':  'Energy Consumed (Joules)',
    }
    renamed_cols = [column_mapping.get(c, c) for c in feature_cols]
    X_test = pd.DataFrame(X_test, columns=renamed_cols)
 
    """
    Predict and compute metrics
    """
    y_pred  = model.predict(X_test)
    report  = classification_report(y_test, y_pred, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred)

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

    """
    1. Confusion matrix
    """
    plt.figure(figsize=(max(6, len(names) * 1.4), max(5, len(names) * 1.2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=names, yticklabels=names)
    plt.title(f'RF {suffix.capitalize()} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'charts/rf/rf_{suffix}_confusion_matrix.png')
    plt.show()

    """
    2. Per-class precision, recall, F1, FPR, FNR
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
    plt.title(f'RF {suffix.capitalize()} - Per-Class Metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'charts/rf/rf_{suffix}_per_class_metrics.png')
    plt.show()

    """
    3: Feature importance (top 10)
    """
    importances = model.feature_importances_
    feat_names  = X_test.columns.tolist() if hasattr(X_test, 'columns') else [f'f{i}' for i in range(len(importances))]
    top_idx     = np.argsort(importances)[::-1][:10]

    plt.figure(figsize=(9, 5))
    plt.barh([feat_names[i] for i in top_idx[::-1]],
             [importances[i] for i in top_idx[::-1]])
    plt.title(f'RF {suffix.capitalize()} - Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(f'charts/rf/rf_{suffix}_feature_importance.png')
    plt.show()

    """
    4. ROC curve (binary only) 
    """
    roc_auc = None
    if cls == 'b':
        y_prob = model.predict_proba(X_test)[:, 0]   # Attack=0
        y_test_bin = (y_test == 0).astype(int)
        fpr_curve, tpr_curve, _ = roc_curve(y_test_bin, y_prob)
        roc_auc = auc(fpr_curve, tpr_curve)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr_curve, tpr_curve, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.title('RF Binary - ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig('charts/rf/rf_binary_roc_curve.png')
        plt.show()

    """
    Save metrics CSV and classification report JSON for comparison with FedAvg
    """
    metrics = {
        'model':           'Random Forest',
        'cls':             cls,
        'accuracy':        round(accuracy_score(y_test, y_pred), 4),
        'macro_precision': round(precision_score(y_test, y_pred, average='macro'), 4),
        'macro_recall':    round(recall_score(y_test, y_pred, average='macro'), 4),
        'macro_f1':        round(f1_score(y_test, y_pred, average='macro'), 4),
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

    pd.DataFrame([metrics]).to_csv(f'charts/rf/rf_{suffix}_metrics.csv', index=False)
    with open(f'charts/rf/rf_{suffix}_report.json', 'w') as f:
        json.dump(report, f, indent=4)

    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f'\nAll charts and metrics saved to charts/rf/')
    print('Done!')