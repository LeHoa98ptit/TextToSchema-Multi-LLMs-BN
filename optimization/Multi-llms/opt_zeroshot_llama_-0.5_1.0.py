import os
import json
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure module 'src' can be imported from the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.finding_the_best_er_model import JointERILPSolverFinal_1

# -------------------------
# Configuration & Paths
# -------------------------
INPUT_FOLDER = os.path.join(project_root, "output/probability/multi-llms/pro_zeroshot_llama_-0.5_1.0")
OUTPUT_FOLDER = os.path.join(project_root, "output/optimization/multi-llms/opt_zeroshot_llama-0.5_1.0-(0.5-0.5-0.5)")

MAX_WORKERS = 20  # Optimize multi-threaded processing

# Create output directory if it does not exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_single_file(filename):
    """
    Processing function for each JSON file
    """
    if not filename.endswith(".json"):
        return False
        
    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    
    # Skip if already processed
    if os.path.exists(output_path):
        return True
        
    start_time = time.time()
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        entities_prob = data.get("entity", {})
        attributes_prob = data.get("attribute", {})
        relationships_prob = data.get("relationship", [])
        
        # -------- FIND THE OPTIMAL ER MODEL WITH ILP --------
        er_model = JointERILPSolverFinal_1(entities_prob, relationships_prob, attributes_prob)
        score, sel_entities, sel_relations, sel_attributes, runtime = er_model.solve(
            lambda_E=0.5,
            lambda_R=0.5,
            lambda_A=0.5,
            min_entities=3  
        )
        
        # Reformat relationships: remove 'description' and 'probability', keep required format
        formatted_relations = []
        for r in sel_relations:
            formatted_relations.append({
                "entity_1": r.get("entity_1"),
                "entity_2": r.get("entity_2"),
                "cardinality": r.get("cardinality", "1:N"),
                "associative_entity": r.get("associative_entity", None)
            })
            
        # Build output JSON
        result_dict = {
            "entity": sel_entities,
            "attribut": sel_attributes,  # Keep the "attribut" key as required
            "relationship": formatted_relations
        }
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(result_dict, f_out, ensure_ascii=False, indent=2)
            
        processing_time = time.time() - start_time
        print(f"[OK] {filename} | Ents: {len(sel_entities)} | Attrs: {sum(len(v) for v in sel_attributes.values())} | Rels: {len(formatted_relations)} | Time: {processing_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"[ERROR] {filename}: {e}")
        return False

def main():
    print(f"Starting ER Model optimization (ILP)...")
    print(f"Source folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}\n")
    
    files_to_process = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".json")]
    print(f"Found {len(files_to_process)} JSON files for ILP optimization.\n")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(process_single_file, files_to_process))

if __name__ == "__main__":
    main()