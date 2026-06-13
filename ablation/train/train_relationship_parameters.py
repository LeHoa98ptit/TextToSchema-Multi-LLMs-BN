"""
Ablation Study — Logistic Regression Parameter Training
=========================================================
Trains a logistic regression on 3 text-based features (x_type removed):
    x_text, x_dep, x_cooccur

Compared to the original train/train_relationship_parameters.py:
  - 'x_type' is excluded from feature_cols.
  - Saved JSON has keys: w_text, w_dep, w_cooccur, bias  (no w_type).

Input:
    ablation/data/relationship_training_data_ablation.csv

Output:
    ablation/train/dataset/relationship_parameters_ablation.json
"""

import os
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# Paths relative to this script's directory (ablation/train/)
_SCRIPT_DIR  = os.path.dirname(__file__)
_ABLATION_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))

DATA_PATH   = os.path.join(_ABLATION_ROOT, 'data',  'relationship_training_data_ablation.csv')
OUTPUT_PATH = os.path.join(_SCRIPT_DIR,   'dataset', 'relationship_parameters_ablation.json')


def main():
    # 1. Load dataset
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Run ablation/train/prepare_relation_data.py first.")
        return

    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    # 2. Features: 3 text-based columns — x_type intentionally omitted
    feature_cols = ['x_text', 'x_dep', 'x_cooccur']
    X = df[feature_cols].fillna(0)
    y = df['label']

    print(f"Dataset: {len(df)} samples | features: {feature_cols}")
    print(f"Label distribution:\n{y.value_counts()}\n")

    # 3. Train / test split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # 4. Optional SMOTE oversampling to handle class imbalance
    if HAS_SMOTE:
        print("\nApplying SMOTE to balance training data ...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    else:
        print("\nWarning: imbalanced-learn not installed — skipping SMOTE.")

    # 5. Grid search over regularisation strength and class weight
    print("Tuning hyperparameters ...")
    param_grid = {
        'C':            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'class_weight': [None, 'balanced'],
    }
    grid_search = GridSearchCV(
        LogisticRegression(solver='lbfgs', max_iter=2000),
        param_grid,
        cv=5,
        scoring='accuracy',
    )
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")

    # 6. Evaluation
    print("\n--- Evaluation on Test Set ---")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # 7. Extract learned weights (3 features only)
    w_text, w_dep, w_cooccur = model.coef_[0]
    bias = model.intercept_[0]

    print("\n--- Learned Parameters (ablation, no x_type) ---")
    print(f"  w_text    = {w_text:.4f}")
    print(f"  w_dep     = {w_dep:.4f}")
    print(f"  w_cooccur = {w_cooccur:.4f}")
    print(f"  bias      = {bias:.4f}")
    print(f"\nFormula: P = sigmoid({w_text:.4f}*x_text + {w_dep:.4f}*x_dep"
          f" + {w_cooccur:.4f}*x_cooccur + {bias:.4f})")

    # 8. Save parameters
    params = {
        "w_text":    float(w_text),
        "w_dep":     float(w_dep),
        "w_cooccur": float(w_cooccur),
        "bias":      float(bias),
        # w_type intentionally absent — ablation variant
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4)

    print(f"\nSaved parameters → {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
