"""
Ablation — Probability Estimation for One-LLM Few-Shot LLaMA
=============================================================
Re-computes entity / attribute / relationship probabilities for
    Output_final/generation/one-llm/one_llm_few_shot_llama
using the ablation modules (no Wikidata x_type).

Output:
    ablation/ablation_one_llm/probability/{id}.json

Usage:
    python ablation/ablation_one_llm/run_probability.py
    python ablation/ablation_one_llm/run_probability.py --workers 8
"""

import os, sys, re, json, time, argparse, hashlib, logging
from concurrent.futures import ThreadPoolExecutor

_SELF_DIR    = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(_SELF_DIR, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if 'CORENLP_HOME' not in os.environ:
    os.environ['CORENLP_HOME'] = ''
_orig_del = os.environ.__class__.__delitem__
def _safe_del(self, key):
    try:
        _orig_del(self, key)
    except KeyError:
        if key != 'CORENLP_HOME':
            raise
os.environ.__class__.__delitem__ = _safe_del

from src.pre_processing import preprocess_text
from ablation.src.entity_processing_ablation import (
    entity_similarity_checker, compute_entity_probabilities, extract_entity_probs,
)
from ablation.src.attribute_processing_ablation import (
    compute_all_attribute_probabilities, extract_attribute_probs,
)
from ablation.src.relationship_processing_ablation import (
    compute_relationship_probabilities_2,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
GEN_FOLDER  = os.path.join(project_root, 'Output_final', 'generation',
                            'one-llm', 'one_llm_few_shot_llama')
PROB_FOLDER = os.path.join(_SELF_DIR, 'probability')
LOG_FOLDER  = os.path.join(_SELF_DIR, 'log')
TXT_ROOT    = os.path.join(project_root, 'dataset', 'Datasets', 'Full-Dataset', 'input')
os.makedirs(PROB_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

BIAS   = 0.5
WEIGHT = 1.0

_preprocess_cache: dict = {}

def _hash(text): return hashlib.md5(text.encode()).hexdigest()

def preprocess_cached(raw):
    h = _hash(raw)
    if h not in _preprocess_cache:
        _preprocess_cache[h] = preprocess_text(raw)
    return _preprocess_cache[h]


def _rel_prob(rel_probs, e1, e2):
    if not e1 or not e2: return 0.0
    e1u, e2u = e1.strip().upper(), e2.strip().upper()
    for item in rel_probs:
        if not isinstance(item, dict): continue
        ie1 = str(item.get('e1', item.get('entity_1', ''))).strip().upper()
        ie2 = str(item.get('e2', item.get('entity_2', ''))).strip().upper()
        if (e1u == ie1 and e2u == ie2) or (e1u == ie2 and e2u == ie1):
            return float(item.get('p', item.get('probability', 0.0)))
    return 0.0


def _attr_prob(attribute_probs, ent, attr):
    if not ent or not attr: return 0.0
    ent_u   = ent.strip().upper()
    attr_cl = re.sub(r'[^a-zA-Z0-9]', '', attr.strip().lower())

    def _val(v):
        try:
            if isinstance(v, (float, int)): return float(v)
            if isinstance(v, str):          return float(v)
            if isinstance(v, dict):
                for k in ('P(Attribute|Entity)', 'probability', 'p'):
                    if k in v: return float(v[k])
        except Exception: pass
        return None

    if isinstance(attribute_probs, list):
        for item in attribute_probs:
            if not isinstance(item, dict): continue
            ie  = str(item.get('Entity', item.get('entity', ''))).strip().upper()
            ia  = re.sub(r'[^a-zA-Z0-9]', '', str(item.get('Attribute', item.get('attribute', ''))).lower())
            if ie == ent_u and ia == attr_cl:
                v = _val(item)
                if v is not None: return v
    elif isinstance(attribute_probs, dict):
        ent_data = attribute_probs.get(ent_u) or attribute_probs.get(ent)
        if isinstance(ent_data, dict):
            for ak, av in ent_data.items():
                if re.sub(r'[^a-zA-Z0-9]', '', str(ak).lower()) == attr_cl:
                    v = _val(av)
                    if v is not None: return v
        elif isinstance(ent_data, list):
            # extract_attribute_probs returns {entity: [(attr_name, prob), ...]}
            for item in ent_data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    if re.sub(r'[^a-zA-Z0-9]', '', str(item[0]).lower()) == attr_cl:
                        v = _val(item[1])
                        if v is not None: return v
    return 0.0


def build_output(gen_data, entities, attributes, relationships,
                 entity_probs, attribute_probs, rel_probs):
    out_entities      = {str(e).upper(): entity_probs.get(str(e).upper(),
                          entity_probs.get(str(e), 0.5))
                         for e in entities}
    out_attributes    = {}
    for ent in entities:
        ent_u = str(ent).upper()
        ent_attrs = (attributes.get(ent) or attributes.get(ent_u)
                     or attributes.get(ent.lower()) or [])
        if isinstance(ent_attrs, dict): ent_attrs = list(ent_attrs.keys())
        if not ent_attrs: continue
        out_attributes[ent_u] = {}
        for attr in ent_attrs:
            p = _attr_prob(attribute_probs, ent, attr)
            if p <= 0.0: p = 0.5
            out_attributes[ent_u][str(attr)] = p

    out_relationships = []
    for rel in relationships:
        e1   = str(rel.get('entity_1', rel.get('e1', ''))).strip().upper()
        e2   = str(rel.get('entity_2', rel.get('e2', ''))).strip().upper()
        prob = _rel_prob(rel_probs, e1, e2)
        if prob <= 0.0: prob = 0.5
        card = str(rel.get('cardinality', '')).upper()

        if card in ('N:M', 'M:N'):
            assoc_info = rel.get('associative_entity')
            if assoc_info and isinstance(assoc_info, dict) and assoc_info.get('name'):
                assoc_name  = str(assoc_info['name']).strip().upper()
                assoc_attrs = assoc_info.get('attributes', [])
            else:
                assoc_name  = f'ASSOC_{e1}_{e2}'.upper()
                assoc_attrs = []

            if assoc_name not in out_entities:
                out_entities[assoc_name] = prob
            else:
                out_entities[assoc_name] = max(out_entities[assoc_name], prob)
            if assoc_name not in out_attributes:
                out_attributes[assoc_name] = {}
            for key_name in [f'{assoc_name.lower()}_id',
                              f'{e1.lower()}_id', f'{e2.lower()}_id'] + assoc_attrs:
                prev = out_attributes[assoc_name].get(key_name, 0.0)
                out_attributes[assoc_name][key_name] = max(prev, prob)

            for new_e1, new_e2 in [(e1, assoc_name), (e2, assoc_name)]:
                new_rel = {'entity_1': new_e1, 'entity_2': new_e2,
                           'cardinality': '1:N', 'associative_entity': None,
                           'probability': prob}
                merged = False
                for r in out_relationships:
                    if r['entity_1'] == new_e1 and r['entity_2'] == new_e2:
                        r['probability'] = max(r['probability'], prob)
                        merged = True; break
                if not merged:
                    out_relationships.append(new_rel)
        else:
            rel_copy = dict(rel)
            rel_copy['probability'] = prob
            out_relationships.append(rel_copy)

    out = dict(gen_data)
    out['entity']       = out_entities
    out['attribute']    = out_attributes
    out['relationship'] = out_relationships
    out.pop('attribut', None)
    return out


def process_file(args):
    filename, gen_folder, out_folder = args
    out_path = os.path.join(out_folder, filename)
    if os.path.exists(out_path):
        return True
    try:
        t0 = time.time()
        txt_path  = os.path.join(TXT_ROOT, filename.replace('.json', '.txt'))
        json_path = os.path.join(gen_folder, filename)
        if not os.path.exists(txt_path):
            print(f'  [SKIP] No text file for {filename}')
            return False

        raw_text = open(txt_path,  'r', encoding='utf-8').read().strip()
        gen_data = json.load(open(json_path, 'r', encoding='utf-8'))
        processed_text = preprocess_cached(raw_text)

        entities      = gen_data.get('entity', [])
        attributes    = gen_data.get('attribute', gen_data.get('attribut', {}))
        relationships = gen_data.get('relationship', [])

        entity_probs    = extract_entity_probs(
            compute_entity_probabilities(
                entity_similarity_checker(processed_text, entities),
                bias=BIAS, weight=WEIGHT))
        attribute_probs = extract_attribute_probs(
            compute_all_attribute_probabilities(
                attributes, bias=BIAS, weight=WEIGHT,
                processed_text=processed_text))
        rel_probs = compute_relationship_probabilities_2(
            json.dumps(relationships), processed_text,
            bias=BIAS, weight=WEIGHT)

        out_data = build_output(gen_data, entities, attributes, relationships,
                                entity_probs, attribute_probs, rel_probs)
        out_data['processing_time'] = round(time.time() - t0, 2)

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f'  [OK] {filename}  ({out_data["processing_time"]:.1f}s)')
        return True
    except Exception as e:
        print(f'  [ERR] {filename}: {e}')
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    json_files = sorted(f for f in os.listdir(GEN_FOLDER)
                        if f.endswith('.json') and not f.startswith('.'))
    print(f'Input  : {GEN_FOLDER}')
    print(f'Output : {PROB_FOLDER}')
    print(f'Files  : {len(json_files)}  |  workers={args.workers}\n')

    task_args = [(fn, GEN_FOLDER, PROB_FOLDER) for fn in json_files]
    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(process_file, task_args):
            if result: ok   += 1
            else:      fail += 1

    print(f'\nDone: {ok} OK, {fail} failed — {time.time()-t0:.1f}s')
    print(f'Output → {PROB_FOLDER}')


if __name__ == '__main__':
    main()
