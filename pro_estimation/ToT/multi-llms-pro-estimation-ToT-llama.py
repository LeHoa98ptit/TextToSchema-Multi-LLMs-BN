import os
import json
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
import hashlib
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
print("OK")

# Ensure module 'src' can be imported from the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Prevent KeyError: 'CORENLP_HOME' when OpenIE is garbage collected
if 'CORENLP_HOME' not in os.environ:
    os.environ['CORENLP_HOME'] = ''

# Monkey-patch os._Environ to safely handle 'CORENLP_HOME' deletion
# so that multiple StanfordOpenIE instances don't raise KeyError on exit.
_orig_delitem = os.environ.__class__.__delitem__
def safe_environ_delitem(self, key):
    try:
        _orig_delitem(self, key)
    except KeyError:
        if key != 'CORENLP_HOME':
            raise
os.environ.__class__.__delitem__ = safe_environ_delitem

from src.pre_processing import preprocess_text
from src.entity_processing import (
    entity_similarity_checker,
    compute_entity_probabilities,
    extract_entity_probs
)
from src.attribute_processing import (
    compute_all_attribute_probabilities,
    extract_attribute_probs
)
from src.relationship_processing import compute_relationship_probabilities_2

# ===== OPTIMIZATION: GLOBAL MODEL LOADING =====
# Load model once, reuse across all files
GLOBAL_MODELS = {}
PREPROCESS_CACHE = {}  # Cache preprocessing results
ENTITY_SIMILARITY_CACHE = {}  # Cache entity similarity results

# -------------------------
# Configuration & Paths
# -------------------------
INPUT_TXT_FOLDER = os.path.join(project_root, "dataset/Datasets/Full-Dataset/input")
INPUT_JSON_FOLDER = os.path.join(project_root, "output/generation/prompt_ToT_llama")
OUTPUT_FOLDER = os.path.join(project_root, "output/probability/ToT/pro_ToT_llama_1.0_3.0")

BIAS = 1.0
WEIGHT = 3.0
MAX_WORKERS = 20  # OPTIMIZED: Increased from 5 to 20 (adjust to CPU cores)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def init_global_models():
    """OPTIMIZED: Initialize models once at program startup"""
    global GLOBAL_MODELS
    print("[INIT] Loading embedding models globally (1 time only)...")
    
    # Load embedding model if needed
    try:
        from sentence_transformers import SentenceTransformer
        GLOBAL_MODELS['embedding_model'] = SentenceTransformer("all-MiniLM-L6-v2")
        print("[INIT] ✓ Embedding model loaded")
    except Exception as e:
        print(f"[WARN] Embedding model not loaded: {e}")
    
    print("[INIT] Models initialized successfully!\n")

def get_text_hash(text):
    """Compute text hash for caching"""
    return hashlib.md5(text.encode()).hexdigest()

def preprocess_text_cached(raw_text):
    """OPTIMIZED: Preprocessing with caching"""
    text_hash = get_text_hash(raw_text)
    
    if text_hash in PREPROCESS_CACHE:
        return PREPROCESS_CACHE[text_hash]
    
    processed = preprocess_text(raw_text)
    PREPROCESS_CACHE[text_hash] = processed
    
    return processed

def get_rel_prob(rel_probs_data, e1, e2):
    """Helper to map probability from the result list of compute_relationship_probabilities_2."""
    if not e1 or not e2: 
        return 0.0
    e1_str, e2_str = str(e1).strip().upper(), str(e2).strip().upper()
    
    if isinstance(rel_probs_data, list):
        for item in rel_probs_data:
            if isinstance(item, dict):
                # Cover both entity_1 and e1 key formats
                i_e1 = str(item.get("entity_1", item.get("e1", ""))).strip().upper()
                i_e2 = str(item.get("entity_2", item.get("e2", ""))).strip().upper()
                if (e1_str in i_e1 and e2_str in i_e2) or (e1_str in i_e2 and e2_str in i_e1):
                    return float(item.get("probability", item.get("p", 0.0)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                str_item = str(item).upper()
                if e1_str in str_item and e2_str in str_item:
                    try: return float(item[-1])
                    except: pass
    elif isinstance(rel_probs_data, dict):
        for k, v in rel_probs_data.items():
            if e1_str in str(k).upper() and e2_str in str(k).upper():
                return float(v)
    return 0.0

def get_attr_prob(attribute_probs, ent, attr):
    """Helper to map probability from attribute_probs, supports both flat and nested dict/list."""
    if not ent or not attr:
        return 0.0
    
    ent_str = str(ent).strip().upper()
    # Remove all special characters, spaces, underscores for exact matching
    attr_clean = re.sub(r'[^a-zA-Z0-9]', '', str(attr).strip().lower())
    
    def extract_val(v):
        try:
            if isinstance(v, (float, int)): return float(v)
            if isinstance(v, str): return float(v)
            if isinstance(v, dict):
                for pk in ["P(Attribute|Entity)", "probability", "p", "prob"]:
                    if pk in v: return float(v[pk])
        except: pass
        return None

    if isinstance(attribute_probs, list):
        for item in attribute_probs:
            if isinstance(item, dict):
                i_ent = str(item.get("Entity", item.get("entity", ""))).strip().upper()
                i_attr = re.sub(r'[^a-zA-Z0-9]', '', str(item.get("Attribute", item.get("attribute", ""))).strip().lower())
                if i_ent == ent_str and i_attr == attr_clean:
                    val = extract_val(item)
                    if val is not None: return val
                    
    elif isinstance(attribute_probs, dict):
        for k, v in attribute_probs.items():
            # 1. Handle tuple keys: ("PLAYER", "player id")
            if isinstance(k, tuple) and len(k) >= 2:
                k_ent = str(k[0]).strip().upper()
                k_attr_clean = re.sub(r'[^a-zA-Z0-9]', '', str(k[1]).lower())
                if k_ent == ent_str and k_attr_clean == attr_clean:
                    val = extract_val(v)
                    if val is not None: return val
                continue
                
            k_str = str(k)
            
            # 2. Handle nested dict/list
            if k_str.strip().upper() == ent_str:
                if isinstance(v, dict):
                    for ak, av in v.items():
                        ak_clean = re.sub(r'[^a-zA-Z0-9]', '', str(ak).lower())
                        if ak_clean == attr_clean:
                            val = extract_val(av)
                            if val is not None: return val
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            ak_clean = re.sub(r'[^a-zA-Z0-9]', '', str(item[0]).lower())
                            if ak_clean == attr_clean:
                                val = extract_val(item[1])
                                if val is not None: return val
                                
            # 3. Handle flat string keys
            k_upper = k_str.upper()
            if ent_str in k_upper:
                remainder = k_upper.replace(ent_str, "").lower()
                rem_clean = re.sub(r'[^a-zA-Z0-9]', '', remainder)
                if rem_clean == attr_clean:
                    val = extract_val(v)
                    if val is not None: return val
                    
            # 4. Fallback: match by attribute name only
            k_clean = re.sub(r'[^a-zA-Z0-9]', '', k_str.lower())
            if k_clean == attr_clean:
                val = extract_val(v)
                if val is not None: return val

    return 0.0

def build_output_data(gen_data, entities, attributes, relationships, 
                     entity_probs, attribute_probs, rel_probs):
    """OPTIMIZED: Pre-allocate and build output data centrally"""
    
    out_entities = {ent: entity_probs.get(ent, 0.0) for ent in entities}
    
    out_attributes = {}
    for ent, attrs in attributes.items():
        out_attributes[ent] = {attr: get_attr_prob(attribute_probs, ent, attr) for attr in attrs}
        
    out_relationships = []
    
    for rel in relationships:
        e1, e2 = rel.get("entity_1"), rel.get("entity_2")
        prob = get_rel_prob(rel_probs, e1, e2)
        
        # Identify N:M relationship to split into associative entity
        card = str(rel.get("cardinality", "")).upper()
        if card in ["N:M", "M:N"]:
            # Use associative_entity suggested by LLM (if available)
            assoc_info = rel.get("associative_entity")
            if assoc_info and isinstance(assoc_info, dict) and assoc_info.get("name"):
                assoc_name = str(assoc_info.get("name")).strip().upper()
                assoc_attrs = assoc_info.get("attributes", [])
            else:
                assoc_name = f"ASSOC_{e1}_{e2}".upper()
                assoc_attrs = []
                
            # Merge Entity Probability if it already exists
            if assoc_name not in out_entities:
                out_entities[assoc_name] = prob
            else:
                out_entities[assoc_name] = max(out_entities[assoc_name], prob)
            
            # Initialize attribute dict if not yet created
            if assoc_name not in out_attributes:
                out_attributes[assoc_name] = {}
            
            # Set primary key name for Associative Entity based on its own name rather than joining e1_e2
            pk_name = f"{assoc_name.lower()}_id"
            if pk_name not in out_attributes[assoc_name]:
                out_attributes[assoc_name][pk_name] = prob
            else:
                out_attributes[assoc_name][pk_name] = max(out_attributes[assoc_name][pk_name], prob)
            
            # Add foreign keys and merge if duplicated
            fk1, fk2 = f"{e1.lower()}_id", f"{e2.lower()}_id"
            if fk1 not in out_attributes[assoc_name]:
                out_attributes[assoc_name][fk1] = prob
            else:
                out_attributes[assoc_name][fk1] = max(out_attributes[assoc_name][fk1], prob)
                
            if fk2 not in out_attributes[assoc_name]:
                out_attributes[assoc_name][fk2] = prob
            else:
                out_attributes[assoc_name][fk2] = max(out_attributes[assoc_name][fk2], prob)
            
            # Merge existing attributes from JSON
            for a in assoc_attrs:
                if a not in out_attributes[assoc_name]:
                    out_attributes[assoc_name][a] = prob
                else:
                    out_attributes[assoc_name][a] = max(out_attributes[assoc_name][a], prob)
                    
            # Split into 2 × 1:N relationships and merge to avoid duplicates
            for new_rel in [
                {
                    "entity_1": e1,
                    "entity_2": assoc_name,
                    "description": f"One {e1} can be associated with many {assoc_name}.",
                    "cardinality": "1:N",
                    "associative_entity": None,
                    "probability": prob
                },
                {
                    "entity_1": e2,
                    "entity_2": assoc_name,
                    "description": f"One {e2} can be associated with many {assoc_name}.",
                    "cardinality": "1:N",
                    "associative_entity": None,
                    "probability": prob
                }
            ]:
                merged = False
                for r in out_relationships:
                    if r.get("entity_1") == new_rel["entity_1"] and r.get("entity_2") == new_rel["entity_2"]:
                        r["probability"] = max(r.get("probability", 0.0), new_rel["probability"])
                        merged = True
                        break
                if not merged:
                    out_relationships.append(new_rel)
        else:
            # For other relationship types (1:1, 1:N), keep a copy as-is
            rel_copy = dict(rel)
            rel_copy["probability"] = prob
            out_relationships.append(rel_copy)
    
    out_data = dict(gen_data)
    out_data["entity"] = out_entities
    if "attribut" in out_data:
        del out_data["attribut"]
    out_data["attribute"] = out_attributes
    out_data["relationship"] = out_relationships
    
    return out_data

def process_single_file(filename):
    """OPTIMIZED: Reduced overhead, caching preprocessing"""
    try:
        txt_filename = filename.replace(".json", ".txt")
        txt_path = os.path.join(INPUT_TXT_FOLDER, txt_filename)
        json_path = os.path.join(INPUT_JSON_FOLDER, filename)
        
        if not os.path.exists(txt_path):
            print(f"[SKIP] Missing text file for {filename}")
            return False
            
        # Read files (still sequential, can be made async later)
        with open(txt_path, 'r', encoding='utf-8') as f:
            raw_text = f.read().strip()
            
        with open(json_path, 'r', encoding='utf-8') as f:
            gen_data = json.load(f)
            
        # OPTIMIZED: Use preprocessing cache
        processed_text = preprocess_text_cached(raw_text)
        
        entities = gen_data.get("entity", [])
        attributes = gen_data.get("attribut", gen_data.get("attribute", {}))
        relationships = gen_data.get("relationship", [])
        
        # 1. Compute Entity Probabilities
        entity_confidence = entity_similarity_checker(processed_text, entities)
        entity_probs = extract_entity_probs(compute_entity_probabilities(entity_confidence, bias=BIAS, weight=WEIGHT))
        
        # 2. Compute Attribute Probabilities
        attribute_probs = extract_attribute_probs(compute_all_attribute_probabilities(attributes, bias=BIAS, weight=WEIGHT))
        
        # 3. Compute Relationship Probabilities
        rel_probs = compute_relationship_probabilities_2(json.dumps(relationships), processed_text, bias=BIAS, weight=WEIGHT)
        
        # OPTIMIZED: Build output data centrally
        out_data = build_output_data(gen_data, entities, attributes, relationships, 
                                    entity_probs, attribute_probs, rel_probs)
        
        out_path = os.path.join(OUTPUT_FOLDER, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
            
        print(f"[OK] {filename}")
        return True
    except Exception as e:
        print(f"[ERROR] {filename}: {str(e)[:50]}")
        return False

def filter_files_by_range(json_files, min_idx=251, max_idx=500):
    """OPTIMIZED: Filter files instead of applying regex one by one"""
    filtered = []
    for filename in json_files:
        try:
            # Extract number from filename
            nums = re.findall(r'\d+', filename)
            if nums:
                idx = int(nums[-1])
                if min_idx <= idx <= max_idx:
                    filtered.append(filename)
        except:
            pass
    return filtered

def main():
    # OPTIMIZED: Initialize global models
    init_global_models()
    
    start_time_total = time.time()
    processed_count = 0

    print(f"Reading JSONs from: {INPUT_JSON_FOLDER}")
    print(f"Reading TXTs from: {INPUT_TXT_FOLDER}")
    print(f"Saving to: {OUTPUT_FOLDER}\n")
    print(f"MAX_WORKERS: {MAX_WORKERS} (OPTIMIZED)")
    print("=" * 70)
    
    json_files = [f for f in os.listdir(INPUT_JSON_FOLDER) if f.endswith(".json")]
    print(f"Found {len(json_files)} JSON files total")
    
    # OPTIMIZED: Use filter function with dataset range
    files_to_process = filter_files_by_range(json_files, min_idx=251, max_idx=500)
    print(f"Filtered to {len(files_to_process)} files to process")
    print(f"Starting multi-threading with {MAX_WORKERS} workers...\n")

    # OPTIMIZED: Increased max_workers from 5 to 20
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Use .map() instead of dict comprehension
        for result in executor.map(process_single_file, files_to_process):
            if result:
                processed_count += 1

    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    avg_time = total_time / processed_count if processed_count > 0 else 0
    files_per_hour = (processed_count / total_time * 3600) if total_time > 0 else 0
    
    report_folder = os.path.join(project_root, "output/cost")
    os.makedirs(report_folder, exist_ok=True)
    report_path = os.path.join(report_folder, "cost_report_pro_estimation_ToT_llama_1.0_3.0.txt")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(f"--- Timestamp: {timestamp} ---\n")
        f.write(f"Total Files Processed: {processed_count}\n")
        f.write(f"Total Calculation Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        f.write(f"Average Time per File: {avg_time:.2f} seconds\n")
        f.write(f"Files per Hour: {files_per_hour:.1f}\n")
        f.write(f"Max Workers: {MAX_WORKERS}\n")
        f.write(f"Cache Size: {len(PREPROCESS_CACHE)} entries\n")
        f.write("==================================================\n")
        
    print("\n" + "=" * 70)
    print(f"Processing completed!")
    print(f"Statistics:")
    print(f"   - Files processed: {processed_count}")
    print(f"   - Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"   - Avg per file: {avg_time:.2f}s")
    print(f"   - Speed: {files_per_hour:.1f} files/hour")
    print(f"   - Cache hits: {len(PREPROCESS_CACHE)} (preprocessing)")
    print(f"Report saved to: {report_path}\n")

if __name__ == "__main__":
    main()