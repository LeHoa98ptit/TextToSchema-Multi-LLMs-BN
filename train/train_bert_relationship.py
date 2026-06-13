"""
Train a relationship classifier using SBERT entity embeddings + hand-crafted features.
Architecture: SBERT encodes entity names → feature vector → MLPClassifier
No fine-tuning required — uses frozen SBERT embeddings.
Output: train/bert_model/relationship_mlp_model.pkl
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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

MODEL_DIR  = os.path.join(project_root, "train", "bert_model")
MODEL_PATH = os.path.join(MODEL_DIR, "relationship_mlp_model.pkl")
DATA_PATH  = os.path.join(project_root, "train", "dataset", "relationship_training_data.csv")


def clean_name(s):
    if not s:
        return ""
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return s.replace('_', ' ').replace('-', ' ').lower().strip()


def build_features(df, sbert):
    """Build feature matrix from entity names + hand-crafted features."""
    e1_names = [clean_name(str(e)) for e in df["entity_1"]]
    e2_names = [clean_name(str(e)) for e in df["entity_2"]]

    print("Encoding entity names with SBERT...")
    all_names = list(set(e1_names + e2_names))
    emb_map = {name: emb for name, emb in zip(all_names, sbert.encode(all_names, batch_size=64, show_progress_bar=True))}

    emb_e1 = np.array([emb_map[n] for n in e1_names])   # (N, 384)
    emb_e2 = np.array([emb_map[n] for n in e2_names])   # (N, 384)

    # Semantic interaction features
    emb_diff   = np.abs(emb_e1 - emb_e2)                # element-wise diff (N, 384)
    emb_prod   = emb_e1 * emb_e2                         # element-wise product (N, 384)
    cos_sim    = (emb_e1 * emb_e2).sum(axis=1, keepdims=True) / (
        np.linalg.norm(emb_e1, axis=1, keepdims=True) *
        np.linalg.norm(emb_e2, axis=1, keepdims=True) + 1e-8
    )                                                    # (N, 1)

    # Hand-crafted features
    hc = df[["x_text", "x_dep", "x_type", "x_cooccur"]].fillna(0).values  # (N, 4)

    # Concatenate all features
    X = np.concatenate([emb_diff, emb_prod, cos_sim, hc], axis=1)
    print(f"Feature matrix shape: {X.shape}")
    return X


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    y  = df["label"].values
    print(f"Dataset: {len(df)} samples | pos={y.sum()} | neg={(1-y).sum()}")

    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    X = build_features(df, sbert)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if HAS_SMOTE:
        print("Applying SMOTE...")
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    print("Training MLP classifier...")
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
        ))
    ])
    pipeline.fit(X_train, y_train)

    print("\n--- Evaluation on Test Set ---")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to: {MODEL_PATH}")
    print("Feature dim:", X.shape[1])


if __name__ == "__main__":
    main()
