"""
Ablation Study — Data Preparation
==================================
Generates the relationship training dataset WITHOUT the Wikidata x_type feature.

Compared to the original train/prepare_relation_data.py:
  - compute_x_type is NOT imported and NOT computed.
  - The saved CSV has columns: exercise_id, entity_1, entity_2,
    x_text, x_dep, x_cooccur, label  (no x_type column).

Output:
    ablation/data/relationship_training_data_ablation.csv

Source exercises (inputs + reference annotations):
    dataset/Datasets/Full-Dataset/input/
    dataset/Datasets/Full-Dataset/Reference/
"""

import os
import sys
import json
import itertools
import re
import pandas as pd
import spacy

# Allow importing from project root (for ablation/src modules)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from ablation.src.relationship_processing_ablation import (
    compute_x_text,
    compute_x_dep,
    compute_x_cooccur,
)

nlp = spacy.load("en_core_web_sm")


def clean_name(s: str) -> str:
    """Normalise entity name: CamelCase / snake_case → lower-cased words."""
    if not s:
        return ""
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return s.replace('_', ' ').replace('-', ' ').lower().strip()


def get_pair_context(doc, e1_str: str, e2_str: str) -> str:
    """
    Return the sentence in the parsed document that mentions both entities.
    Falls back to a generic template when no such sentence is found.
    This mirrors the logic used at inference time so features are consistent.
    """
    e1_lower = e1_str.lower()
    e2_lower = e2_str.lower()
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if e1_lower in sent_text and e2_lower in sent_text:
            return sent.text.strip()
    return f"A {e1_str.lower()} can be associated with one or more {e2_str.lower()}."


def main():
    INPUT_DIR = os.path.join(project_root, "dataset", "Datasets", "Full-Dataset", "input")
    REF_DIR   = os.path.join(project_root, "dataset", "Datasets", "Full-Dataset", "Reference")
    OUTPUT_CSV = os.path.join(project_root, "ablation", "data", "relationship_training_data_ablation.csv")

    data_rows = []

    print("Preparing ablation training data (exercises 1–250, no x_type) ...")
    for i in range(1, 251):
        # Locate input text file
        input_file = os.path.join(INPUT_DIR, f"exercise{i}.txt")
        if not os.path.exists(input_file):
            input_file = os.path.join(INPUT_DIR, f"{i}.txt")
            if not os.path.exists(input_file):
                print(f"  Skipping {i} — input not found.")
                continue

        # Locate reference annotation file
        ref_file = os.path.join(REF_DIR, f"exercise{i}-baseline.txt")
        if not os.path.exists(ref_file):
            print(f"  Skipping {i} — reference not found.")
            continue

        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        with open(ref_file, 'r', encoding='utf-8') as f:
            try:
                ref_data = json.load(f)
            except json.JSONDecodeError:
                print(f"  Skipping {i} — invalid JSON in reference.")
                continue

        doc = nlp(text)
        entities      = ref_data.get("entity", [])
        relationships = ref_data.get("relationship", [])

        # Build set of ground-truth (positive) entity pairs
        true_relations = set()
        for r in relationships:
            e1, e2 = r.get("entity_1"), r.get("entity_2")
            if e1 and e2:
                true_relations.add(tuple(sorted([str(e1), str(e2)])))

        # Generate all entity pairs; label = 1 if in ground truth, else 0
        for e1, e2 in itertools.combinations(entities, 2):
            e1_str = clean_name(str(e1))
            e2_str = clean_name(str(e2))

            label   = 1 if tuple(sorted([str(e1), str(e2)])) in true_relations else 0
            context = get_pair_context(doc, e1_str, e2_str)

            # Compute only text-based features (Wikidata x_type intentionally omitted)
            x_text    = compute_x_text(context, text)
            x_dep     = compute_x_dep(context, e1_str, e2_str)
            x_cooccur = compute_x_cooccur(text, e1_str, e2_str)

            data_rows.append({
                "exercise_id": i,
                "entity_1":   e1,
                "entity_2":   e2,
                "x_text":     x_text,
                "x_dep":      x_dep,
                "x_cooccur":  x_cooccur,
                "label":      label,
            })

        print(f"  Completed exercise {i}.")

    df = pd.DataFrame(data_rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(df)} rows → {OUTPUT_CSV}")
    print("Label distribution:")
    print(df['label'].value_counts())


if __name__ == '__main__':
    main()
