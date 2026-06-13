import os
import json
import itertools
import re
import pandas as pd
import spacy
import sys

# Add project path to sys.path to import files from src/
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.relationship_processing import compute_x_type, compute_x_text, compute_x_dep, compute_x_cooccur

nlp = spacy.load("en_core_web_sm")

def clean_name(s):
    if not s: return ""
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return s.replace('_', ' ').replace('-', ' ').lower().strip()

def get_pair_context(doc, e1_str, e2_str):
    """
    Finds the sentence containing both entities in the original text.
    This sentence acts as the 'description' passed to compute_x_text and compute_x_dep,
    ensuring the feature calculation logic matches the inference system.
    """
    e1_lower = e1_str.lower()
    e2_lower = e2_str.lower()
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if e1_lower in sent_text and e2_lower in sent_text:
            return sent.text.strip()
            
    # Use an LLM-simulating template as a fallback if not found in the original text.
    return f"A {e1_str.lower()} can be associated with one or more {e2_str.lower()}."

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    INPUT_DIR = os.path.join(project_root, "dataset/Datasets/Full-Dataset/input")
    REF_DIR = os.path.join(project_root, "dataset/Datasets/Full-Dataset/Reference")
    
    data_rows = []
    
    print("Starting to process 250 data files to create the training set...")
    for i in range(1, 251):
        input_file = os.path.join(INPUT_DIR, f"exercise{i}.txt")
        if not os.path.exists(input_file):
            input_file = os.path.join(INPUT_DIR, f"{i}.txt")
            if not os.path.exists(input_file):
                print(f"Skipping dataset {i} - Input file not found.")
                continue
                
        ref_file = os.path.join(REF_DIR, f"exercise{i}-baseline.txt")
        if not os.path.exists(ref_file):
            print(f"Skipping dataset {i} - Reference file not found.")
            continue
            
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            
        with open(ref_file, 'r', encoding='utf-8') as f:
            try:
                ref_data = json.load(f)
            except json.JSONDecodeError:
                continue
                
        doc = nlp(text)
        entities = ref_data.get("entity", [])
        relationships = ref_data.get("relationship", [])
        
        # Create a set of TRUE relationship pairs (label = 1)
        true_relations = set()
        for r in relationships:
            e1, e2 = r.get("entity_1"), r.get("entity_2")
            if e1 and e2:
                true_relations.add(tuple(sorted([str(e1), str(e2)])))
                
        # Generate all possible combinations of entity pairs
        for e1, e2 in itertools.combinations(entities, 2):
            e1_str = clean_name(str(e1))
            e2_str = clean_name(str(e2))
            
            is_relation = tuple(sorted([str(e1), str(e2)])) in true_relations
            label = 1 if is_relation else 0
            
            # Get the representative context for the entity pair
            context = get_pair_context(doc, e1_str, e2_str)
            
            # USE THE EXACT FUNCTIONS FROM SRC/RELATIONSHIP_PROCESSING.PY
            x_text    = compute_x_text(context, text)
            x_dep     = compute_x_dep(context, e1_str, e2_str)
            x_type    = compute_x_type(e1_str, e2_str)
            x_cooccur = compute_x_cooccur(text, e1_str, e2_str)

            data_rows.append({
                "exercise_id": i,
                "entity_1": e1,
                "entity_2": e2,
                "x_text":    x_text,
                "x_dep":     x_dep,
                "x_type":    x_type,
                "x_cooccur": x_cooccur,
                "label": label
            })
            
        print(f"Completed dataset {i}...")
            
    df = pd.DataFrame(data_rows)
    output_csv = os.path.join(project_root, "train/dataset/relationship_training_data.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\nDone! Created file {output_csv} with {len(df)} data rows.")
    print("Label distribution:")
    print(df['label'].value_counts())

if __name__ == '__main__':
    main()