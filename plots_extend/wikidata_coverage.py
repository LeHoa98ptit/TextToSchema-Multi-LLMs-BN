"""
Analyze Wikidata coverage over the dataset.
- % entities with a Wikidata QID hit
- % exercises with ≥1 entity hit
- % relationship pairs where both entities have QID hits
"""
import os, json, time, requests
from functools import lru_cache
from collections import defaultdict

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_FOLDER = os.path.join(ROOT, "output/generation/multi-llms/few-shot-llama")
CACHE_FILE = os.path.join(ROOT, "plots_extend/wikidata_qid_cache.json")

WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "ERSchemaResearch/1.0 (phd-research; wikidata-coverage-analysis)"}

# ── Load or init QID cache (persist to avoid re-querying) ─────────────────────
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        qid_cache = json.load(f)
    print(f"Loaded {len(qid_cache)} cached QID lookups")
else:
    qid_cache = {}

def clean_label(name: str) -> str:
    """PARKING_SPACE → 'parking space', CamelCase → 'camel case'"""
    import re
    s = name.replace("_", " ").replace("-", " ")
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return s.lower().strip()

SENTINEL = "__MISS__"   # confirmed no result (vs None = not yet queried)

def search_wikidata_qid(label: str) -> str | None:
    """Return QID string if found, else None. Retries up to 3× on error."""
    key = clean_label(label)
    if key in qid_cache:
        v = qid_cache[key]
        return None if v == SENTINEL else v
    params = {
        "action": "wbsearchentities",
        "search": key,
        "language": "en",
        "format": "json",
        "limit": 3,
    }
    qid = None
    for attempt in range(3):
        try:
            r = requests.get(WIKIDATA_SEARCH_URL, params=params,
                             headers=HEADERS, timeout=15)
            hits = r.json().get("search", [])
            qid = hits[0]["id"] if hits else None
            break
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    qid_cache[key] = qid if qid else SENTINEL
    time.sleep(0.2)
    return qid

# ── Collect all entities and relationships from generation output ──────────────
ex_entities   = {}   # num -> list[str]
ex_relations  = {}   # num -> list[(e1, e2)]

for fname in sorted(os.listdir(GEN_FOLDER)):
    if not fname.endswith(".json") or fname.startswith("."): continue
    try:
        num = int(fname.replace(".json", ""))
        with open(os.path.join(GEN_FOLDER, fname)) as f:
            d = json.load(f)
        ents = d.get("entity", [])
        rels = [(r.get("entity_1"), r.get("entity_2"))
                for r in d.get("relationship", [])
                if r.get("entity_1") and r.get("entity_2")]
        ex_entities[num]  = ents
        ex_relations[num] = rels
    except Exception:
        pass

all_entity_names = sorted({e for ents in ex_entities.values() for e in ents})
print(f"\nExercises loaded : {len(ex_entities)}")
print(f"Unique entity names: {len(all_entity_names)}")

# ── Query Wikidata for each unique entity name ─────────────────────────────────
need_query = [e for e in all_entity_names
              if clean_label(e) not in qid_cache or qid_cache[clean_label(e)] == SENTINEL]
print(f"Need to query    : {len(need_query)} (rest from cache)")

for i, name in enumerate(need_query):
    qid = search_wikidata_qid(name)
    if (i + 1) % 50 == 0 or i == len(need_query) - 1:
        hits_so_far = sum(1 for n in need_query[:i+1]
                          if qid_cache.get(n.lower().strip()) is not None)
        print(f"  [{i+1}/{len(need_query)}]  hits={hits_so_far}")
        with open(CACHE_FILE, "w") as f:
            json.dump(qid_cache, f)

# Final cache save
with open(CACHE_FILE, "w") as f:
    json.dump(qid_cache, f)

# ── Compute coverage statistics ───────────────────────────────────────────────
def has_qid(name):
    v = qid_cache.get(clean_label(name))
    return v is not None and v != SENTINEL

# 1. Entity-level coverage
total_ents = sum(len(v) for v in ex_entities.values())
hit_ents   = sum(1 for ents in ex_entities.values() for e in ents if has_qid(e))
print(f"\n── Entity coverage ──────────────────────────────────────")
print(f"  Total entity occurrences : {total_ents}")
print(f"  With Wikidata QID hit    : {hit_ents}  ({100*hit_ents/total_ents:.1f}%)")

# 2. Exercise-level: % exercises with ≥1 entity hit
ex_with_hit = sum(1 for ents in ex_entities.values() if any(has_qid(e) for e in ents))
print(f"\n── Exercise coverage ────────────────────────────────────")
print(f"  Exercises with ≥1 entity hit : {ex_with_hit}/{len(ex_entities)}  ({100*ex_with_hit/len(ex_entities):.1f}%)")

# 3. Relationship pair coverage: both entities have QID
total_pairs  = sum(len(v) for v in ex_relations.values())
hit_pairs    = sum(1 for rels in ex_relations.values()
                   for e1, e2 in rels if has_qid(e1) and has_qid(e2))
partial_pairs = sum(1 for rels in ex_relations.values()
                    for e1, e2 in rels if has_qid(e1) or has_qid(e2))
print(f"\n── Relationship pair coverage ───────────────────────────")
print(f"  Total relationship pairs    : {total_pairs}")
print(f"  Both entities have QID      : {hit_pairs}  ({100*hit_pairs/total_pairs:.1f}%)  ← SPARQL triggered")
print(f"  At least one entity has QID : {partial_pairs}  ({100*partial_pairs/total_pairs:.1f}%)")

# 4. Per-exercise both-hit rate distribution
both_rates = []
for num, rels in ex_relations.items():
    if not rels: continue
    r = sum(1 for e1,e2 in rels if has_qid(e1) and has_qid(e2)) / len(rels)
    both_rates.append(r)
import numpy as np
print(f"\n── Per-exercise SPARQL-trigger rate (among exercises with rels) ──")
print(f"  Mean  : {np.mean(both_rates):.3f}")
print(f"  Median: {np.median(both_rates):.3f}")
print(f"  ≥50%  : {sum(1 for r in both_rates if r>=0.5)}/{len(both_rates)} exercises")
print(f"  100%  : {sum(1 for r in both_rates if r>=1.0)}/{len(both_rates)} exercises")
print(f"   0%   : {sum(1 for r in both_rates if r==0.0)}/{len(both_rates)} exercises")

# ── Save results to file ───────────────────────────────────────────────────────
results = {
    "entity_coverage": {
        "total_occurrences": total_ents,
        "with_qid": hit_ents,
        "pct": round(100 * hit_ents / total_ents, 1),
    },
    "exercise_coverage": {
        "total": len(ex_entities),
        "with_ge1_hit": ex_with_hit,
        "pct": round(100 * ex_with_hit / len(ex_entities), 1),
    },
    "relationship_pair_coverage": {
        "total_pairs": total_pairs,
        "both_qid": hit_pairs,
        "both_qid_pct": round(100 * hit_pairs / total_pairs, 1),
        "at_least_one_qid": partial_pairs,
        "at_least_one_pct": round(100 * partial_pairs / total_pairs, 1),
    },
    "per_exercise_sparql_rate": {
        "n_exercises_with_rels": len(both_rates),
        "mean": round(float(np.mean(both_rates)), 3),
        "median": round(float(np.median(both_rates)), 3),
        "ge50pct": sum(1 for r in both_rates if r >= 0.5),
        "full100pct": sum(1 for r in both_rates if r >= 1.0),
        "zero": sum(1 for r in both_rates if r == 0.0),
    },
}

out_json = os.path.join(ROOT, "plots_extend/wikidata_coverage_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {out_json}")
