"""
Ablation Study — MLP + SBERT Classifier Training
==================================================
Trains the SBERT-based MLP relationship classifier WITHOUT the Wikidata
x_type feature.  The hand-crafted feature vector is now 3-dimensional:
    [x_text, x_dep, x_cooccur]

Compared to the original train/train_bert_relationship.py:
  - 'x_type' column is excluded from the hand-crafted (hc) feature block.
  - The full feature vector becomes:
        [SBERT diff (384), SBERT prod (384), cos_sim (1), hc (3)]  = 772 dims
    vs. the original 773 dims (hc was 4).
  - Model is saved to ablation/train/bert_model/relationship_mlp_model_ablation.pkl
    so it does NOT overwrite the original model.

Input:
    ablation/data/relationship_training_data_ablation.csv

Output:
    ablation/train/bert_model/relationship_mlp_model_ablation.pkl
"""

import os
import sys
import re
import pickle
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

from sentence_transformers import SentenceTransformer

# Paths
_SCRIPT_DIR    = os.path.dirname(__file__)
_ABLATION_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))

DATA_PATH   = os.path.join(_ABLATION_ROOT, 'data',  'relationship_training_data_ablation.csv')
MODEL_DIR   = os.path.join(_SCRIPT_DIR,   'bert_model')
MODEL_PATH  = os.path.join(MODEL_DIR,     'relationship_mlp_model_ablation.pkl')


def clean_name(s: str) -> str:
    """Normalise entity name for SBERT encoding."""
    if not s:
        return ""
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return s.replace('_', ' ').replace('-', ' ').lower().strip()


def build_features(df: pd.DataFrame, sbert: SentenceTransformer) -> np.ndarray:
    """
    Build the feature matrix used at training time.

    Structure (772 dims total):
        diff      (384)  — element-wise absolute difference of SBERT embeddings
        prod      (384)  — element-wise product of SBERT embeddings
        cos_sim     (1)  — cosine similarity between the two embeddings
        hc          (3)  — hand-crafted: [x_text, x_dep, x_cooccur]
                           NOTE: x_type is intentionally omitted (ablation)

    This matches _bert_predict_proba() in relationship_processing_ablation.py,
    so inference features are consistent with training features.
    """
    e1_names = [clean_name(str(e)) for e in df["entity_1"]]
    e2_names = [clean_name(str(e)) for e in df["entity_2"]]

    print("Encoding entity names with SBERT ...")
    all_names = list(set(e1_names + e2_names))
    emb_map = {
        name: emb
        for name, emb in zip(
            all_names,
            sbert.encode(all_names, batch_size=64, show_progress_bar=True),
        )
    }

    emb_e1 = np.array([emb_map[n] for n in e1_names])   # (N, 384)
    emb_e2 = np.array([emb_map[n] for n in e2_names])   # (N, 384)

    # Semantic interaction features
    emb_diff = np.abs(emb_e1 - emb_e2)                  # (N, 384)
    emb_prod = emb_e1 * emb_e2                           # (N, 384)
    cos_sim  = (emb_e1 * emb_e2).sum(axis=1, keepdims=True) / (
        np.linalg.norm(emb_e1, axis=1, keepdims=True) *
        np.linalg.norm(emb_e2, axis=1, keepdims=True) + 1e-8
    )                                                    # (N, 1)

    # Hand-crafted features: 3 columns only (x_type omitted)
    hc = df[["x_text", "x_dep", "x_cooccur"]].fillna(0).values  # (N, 3)

    X = np.concatenate([emb_diff, emb_prod, cos_sim, hc], axis=1)
    print(f"Feature matrix shape: {X.shape}  (expected N × 772)")
    return X


def main():
    # 1. Load ablation dataset (no x_type column)
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Run ablation/train/prepare_relation_data.py first.")
        return

    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    y  = df["label"].values
    print(f"Dataset: {len(df)} samples | pos={y.sum()} | neg={(1-y).sum()}")

    # 2. Build feature matrix
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    X = build_features(df, sbert)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Optional SMOTE oversampling
    if HAS_SMOTE:
        print("Applying SMOTE ...")
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    # 5. Train MLP inside a StandardScaler pipeline
    print("Training MLP classifier ...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            alpha=0.001,
        )),
    ])
    pipeline.fit(X_train, y_train)

    # 6. Evaluation
    print("\n--- Evaluation on Test Set ---")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 7. Save ablation model (separate file — does not overwrite original)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\nAblation model saved → {MODEL_PATH}")
    print(f"Feature dimensionality: {X.shape[1]}")


if __name__ == "__main__":
    main()
