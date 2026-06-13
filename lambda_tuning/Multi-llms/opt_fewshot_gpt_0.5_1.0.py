import os
import json
import time
import sys
from concurrent.futures import ThreadPoolExecutor

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ilp_complexity import JointERILPComplexity

INPUT_FOLDER  = os.path.join(project_root, "output/probability/multi-llms/pro_fewshot_llama_0.5_1.0")
OUTPUT_FOLDER = os.path.join(project_root, "output/lambda_tuning/multi-llms/opt_fewshot_llama_0.5_1.0")
MAX_WORKERS   = 20

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def process_single_file(filename):
    if not filename.endswith(".json"):
        return False

    input_path  = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    if os.path.exists(output_path):
        return True

    start = time.time()
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entities_prob     = data.get("entity", {})
        attributes_prob   = data.get("attribute", {})
        relationships_prob = data.get("relationship", [])

        solver = JointERILPComplexity(entities_prob, relationships_prob, attributes_prob)
        score, sel_entities, sel_relations, sel_attributes, runtime = solver.solve(
            lambda_E=0.603,
            lambda_A=0.455,
            lambda_R=0.688,
            lambda_noattr=1.656,
            lambda_NM=1.996,
            min_entities=3,
        )

        formatted_relations = [
            {
                "entity_1": r["entity_1"],
                "entity_2": r["entity_2"],
                "cardinality": r.get("cardinality", "1:N"),
                "associative_entity": r.get("associative_entity", None),
            }
            for r in sel_relations
        ]

        result = {
            "entity": sel_entities,
            "attribut": sel_attributes,
            "relationship": formatted_relations,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start
        n_attrs = sum(len(v) for v in sel_attributes.values())
        print(f"[OK] {filename} | Ents: {len(sel_entities)} | Attrs: {n_attrs} | Rels: {len(formatted_relations)} | Time: {elapsed:.2f}s")
        return True

    except Exception as e:
        print(f"[ERROR] {filename}: {e}")
        return False


def main():
    print(f"Starting ER optimization (complexity-aware ILP)...")
    print(f"Input : {INPUT_FOLDER}")
    print(f"Output: {OUTPUT_FOLDER}\n")

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".json")]
    print(f"Found {len(files)} JSON files.\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(process_single_file, files))


if __name__ == "__main__":
    main()
