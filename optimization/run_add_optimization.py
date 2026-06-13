"""
ILP optimization cho 18 files trong output/add/probability/.
lambda_E=1.2, lambda_R=-0.5, lambda_A=1.0  (same as opt_fewshot_llama_0.5_1.0-(1.2--0.5-1.0))
Output: output/add/optimization/
"""
import os, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.finding_the_best_er_model import JointERILPSolverFinal_1

INPUT_FOLDER  = os.path.join(project_root, "output/add/probability")
OUTPUT_FOLDER = os.path.join(project_root, "output/add/optimization")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

LAMBDA_E, LAMBDA_R, LAMBDA_A = 1.2, -0.5, 1.0
MIN_ENTITIES = 3


def process_file(filename):
    if not filename.endswith(".json") or filename.startswith("._"):
        return False

    in_path  = os.path.join(INPUT_FOLDER,  filename)
    out_path = os.path.join(OUTPUT_FOLDER, filename)

    start = time.time()
    try:
        with open(in_path, encoding="utf-8") as f:
            data = json.load(f)

        entities_prob      = data.get("entity", {})
        attributes_prob    = data.get("attribute", {})
        relationships_prob = data.get("relationship", [])

        solver = JointERILPSolverFinal_1(entities_prob, relationships_prob, attributes_prob)
        score, sel_entities, sel_relations, sel_attributes, runtime = solver.solve(
            lambda_E=LAMBDA_E,
            lambda_R=LAMBDA_R,
            lambda_A=LAMBDA_A,
            min_entities=MIN_ENTITIES,
            no_isolated=True,
        )

        formatted_rels = [
            {
                "entity_1":          r.get("entity_1"),
                "entity_2":          r.get("entity_2"),
                "cardinality":       r.get("cardinality", "1:N"),
                "associative_entity": r.get("associative_entity"),
            }
            for r in sel_relations
        ]

        result = {
            "entity":       sel_entities,
            "attribut":     sel_attributes,
            "relationship": formatted_rels,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        n_a = sum(len(v) for v in sel_attributes.values())
        # Check isolated
        connected = set()
        for r in formatted_rels:
            connected.add(r["entity_1"]); connected.add(r["entity_2"])
        iso = [e for e in sel_entities if e not in connected]
        iso_str = f" ISO={iso}" if iso else ""

        print(f"[OK] {filename}  E={len(sel_entities)} A={n_a} R={len(formatted_rels)}{iso_str}  ({time.time()-start:.2f}s)")
        return True

    except Exception as e:
        print(f"[ERROR] {filename}: {e}")
        return False


if __name__ == "__main__":
    files = sorted(f for f in os.listdir(INPUT_FOLDER)
                   if f.endswith(".json") and not f.startswith("._"))

    print("=" * 65)
    print(f"ILP Optimization — {len(files)} files")
    print(f"  lambda_E={LAMBDA_E}  lambda_R={LAMBDA_R}  lambda_A={LAMBDA_A}")
    print(f"  Input : {INPUT_FOLDER}")
    print(f"  Output: {OUTPUT_FOLDER}")
    print("=" * 65)

    ok, fail = 0, []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(process_file, f): f for f in files}
        for future in as_completed(futures):
            if future.result():
                ok += 1
            else:
                fail.append(futures[future])

    print("\n" + "=" * 65)
    print(f"Done: {ok}/{len(files)} succeeded")
    if fail:
        print(f"Failed: {fail}")

    # Verification
    print("\n── Verification ──")
    for fname in files:
        p = os.path.join(OUTPUT_FOLDER, fname)
        if not os.path.exists(p):
            print(f"  {fname}: NOT GENERATED")
            continue
        with open(p) as f: d = json.load(f)
        ents = d.get("entity", [])
        attrs = d.get("attribut", {})
        rels  = d.get("relationship", [])
        n_a = sum(len(v) for v in attrs.values())
        connected = {r.get("entity_1") for r in rels} | {r.get("entity_2") for r in rels}
        iso = [e for e in ents if e not in connected]
        flag = ""
        if n_a == 0: flag += " attr=0"
        if len(rels) == 0: flag += " rel=0"
        if iso: flag += f" ISO={iso}"
        print(f"  {fname}: E={len(ents)} A={n_a} R={len(rels)}{flag}")
