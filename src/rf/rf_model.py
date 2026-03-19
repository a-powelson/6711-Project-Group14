"""
Amina Fatma Khan
B00868087
March 18, 2026

Sources: https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/
"""

from preprocessing import load_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder

# Data
df, X, y = load_data()
X_train, X_test, yb_train, yb_test, ym_train, ym_test = train_test_split(
    X, y_binary, y_multi,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)

x_feature = "Dist_To_CH"
y_feature = "Consumed_Energy"

# Binary Classification
rf_binary = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_binary.fit(X_train, yb_train)
yb_pred = rf_binary.predict(X_test)

print("Binary")
print("Accuracy:", accuracy_score(yb_test, yb_pred))
print("Precision:", precision_score(yb_test, yb_pred, pos_label="Attack"))
print("Recall:", recall_score(yb_test, yb_pred, pos_label="Attack"))
print("F1-score:", f1_score(yb_test, yb_pred, pos_label="Attack"))
print("\nClassification Report:\n")
print(classification_report(yb_test, yb_pred))

cm = confusion_matrix(yb_test, yb_pred, labels=["Attack", "Normal"])
print("Confusion Matrix:\n", cm)

# tp = cm[0, 0]
# fn = cm[0, 1]
# fp = cm[1, 0]
# tn = cm[1, 1]
# fpr = fp / (fp + tn)
# print("False Positive Rate:", fpr)

# Binary plot - to do

# Multi Class Classification
rf_multi = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_multi.fit(X_train, ym_train)
ym_pred = rf_multi.predict(X_test)

print("\nMulti-class")
print("Accuracy:", accuracy_score(ym_test, ym_pred))
print("\nClassification Report:\n")
print(classification_report(ym_test, ym_pred))

# Multi class plot - to do
