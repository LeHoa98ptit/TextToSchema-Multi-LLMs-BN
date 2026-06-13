"""
Ablation Study — Probability Estimation Pipeline
=================================================
Re-computes entity / attribute / relationship probabilities for all generation
variants using the ablation modules (no Wikidata x_type).

Differences vs. the original pro_estimation scripts:
  - Attribute:     x = x_text only  (was 0.6*x_text + 0.4*x_type)
  - Relationship:  3 features (x_text, x_dep, x_cooccur) + ablation MLP model
                   (was 4 features including x_type)
  - Entity:        unchanged (entity processing never used Wikidata)

Inputs  (generation outputs already computed):
    Output_final/generation/{variant}/{id}.json  +  dataset/…/input/{id}.txt

Outputs (ablation probability files):
    ablation/probability_estimation_ablation/output/{variant}/{id}.json

Variants processed (10 total):
    multi-llms : few-shot-gpt, few-shot-llama, zero-shot-gpt, zero-shot-llama
    one-llm    : one_llm_few_shot_gpt, one_llm_few_shot_llama,
                 one_llm_zero_shot_gpt, one_llm_zero_shot_llama
    ToT        : prompt_ToT_gpt, prompt_ToT_llama

Logs:
    ablation/probability_estimation_ablation/log/{variant}.txt   (one file per variant)

Usage:
    python ablation/probability_estimation_ablation/run_ablation_probability.py
    python ablation/probability_estimation_ablation/run_ablation_probability.py --variant few-shot-gpt
    python ablation/probability_estimation_ablation/run_ablation_probability.py --workers 10
"""

import os
import sys
import re
import json
import time
import argparse
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# ── Project root ────────────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Prevent KeyError when StanfordOpenIE cleans up CORENLP_HOME
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

# ── Ablation modules (no Wikidata) ───────────────────────────────────────────
from src.pre_processing import preprocess_text
from ablation.src.entity_processing_ablation import (
    entity_similarity_checker,
    compute_entity_probabilities,
    extract_entity_probs,
)
from ablation.src.attribute_processing_ablation import (
    compute_all_attribute_probabilities,
    extract_attribute_probs,
)
from ablation.src.relationship_processing_ablation import (
    compute_relationship_probabilities_2,
)

# ── Paths ────────────────────────────────────────────────────────────────────
_SELF_DIR = os.path.dirname(__file__)
GEN_ROOT  = os.path.join(project_root, 'Output_final', 'generation')
OUT_ROOT  = os.path.join(_SELF_DIR, 'output')
LOG_ROOT  = os.path.join(_SELF_DIR, 'log')
TXT_ROOT  = os.path.join(project_root, 'dataset', 'Datasets', 'Full-Dataset', 'input')

# ── Sigmoid calibration — set via CLI (--bias / --weight) ────────────────────
# Default matches the original best-performing setting: bias=0.5, weight=1.0
BIAS   = 0.5
WEIGHT = 1.0

# ── All generation variants ───────────────────────────────────────────────────
# Each entry: (generation_subfolder,  output_subfolder)
VARIANTS = [
    # multi-llms
    (os.path.join('multi-llms', 'few-shot-gpt'),   os.path.join('multi-llms', 'few-shot-gpt')),
    (os.path.join('multi-llms', 'few-shot-llama'),  os.path.join('multi-llms', 'few-shot-llama')),
    (os.path.join('multi-llms', 'zero-shot-gpt'),   os.path.join('multi-llms', 'zero-shot-gpt')),
    (os.path.join('multi-llms', 'zero-shot-llama'), os.path.join('multi-llms', 'zero-shot-llama')),
    # one-llm
    (os.path.join('one-llm', 'one_llm_few_shot_gpt'),   os.path.join('one-llm', 'one_llm_few_shot_gpt')),
    (os.path.join('one-llm', 'one_llm_few_shot_llama'),  os.path.join('one-llm', 'one_llm_few_shot_llama')),
    (os.path.join('one-llm', 'one_llm_zero_shot_gpt'),   os.path.join('one-llm', 'one_llm_zero_shot_gpt')),
    (os.path.join('one-llm', 'one_llm_zero_shot_llama'), os.path.join('one-llm', 'one_llm_zero_shot_llama')),
    # ToT
    ('prompt_ToT_gpt',   os.path.join('ToT', 'gpt')),
    ('prompt_ToT_llama', os.path.join('ToT', 'llama')),
]

# ── In-process caches ─────────────────────────────────────────────────────────
_preprocess_cache: dict = {}


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def preprocess_cached(raw: str) -> str:
    h = _hash(raw)
    if h not in _preprocess_cache:
        _preprocess_cache[h] = preprocess_text(raw)
    return _preprocess_cache[h]


# ── Helper: look up relationship probability by entity pair ───────────────────
def _rel_prob(rel_probs: list, e1: str, e2: str) -> float:
    if not e1 or not e2:
        return 0.0
    e1u, e2u = e1.strip().upper(), e2.strip().upper()
    for item in rel_probs:
        if not isinstance(item, dict):
            continue
        ie1 = str(item.get('e1', item.get('entity_1', ''))).strip().upper()
        ie2 = str(item.get('e2', item.get('entity_2', ''))).strip().upper()
        if (e1u == ie1 and e2u == ie2) or (e1u == ie2 and e2u == ie1):
            return float(item.get('p', item.get('probability', 0.0)))
    return 0.0


# ── Helper: look up attribute probability ────────────────────────────────────
def _attr_prob(attribute_probs, ent: str, attr: str) -> float:
    if not ent or not attr:
        return 0.0
    ent_u   = ent.strip().upper()
    attr_cl = re.sub(r'[^a-zA-Z0-9]', '', attr.strip().lower())

    def _val(v):
        try:
            if isinstance(v, (float, int)): return float(v)
            if isinstance(v, str):          return float(v)
            if isinstance(v, dict):
                for k in ('P(Attribute|Entity)', 'probability', 'p'):
                    if k in v: return float(v[k])
        except Exception:
            pass
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
            for item in ent_data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    if re.sub(r'[^a-zA-Z0-9]', '', str(item[0]).lower()) == attr_cl:
                        v = _val(item[1])
                        if v is not None: return v
    return 0.0


# ── Build the probability-annotated output dict ───────────────────────────────
def build_output(gen_data: dict, entities: list, attributes: dict,
                 relationships: list, entity_probs: dict,
                 attribute_probs, rel_probs: list) -> dict:
    """
    Annotate the generation dict with probabilities for entities,
    attributes, and relationships.  N:M relationships are expanded into
    two 1:N relationships pointing to an associative entity.
    """
    out_entities   = {ent: entity_probs.get(ent, 0.0) for ent in entities}
    out_attributes = {
        ent: {attr: _attr_prob(attribute_probs, ent, attr) for attr in attrs}
        for ent, attrs in attributes.items()
    }
    out_relationships = []

    for rel in relationships:
        e1   = rel.get('entity_1')
        e2   = rel.get('entity_2')
        prob = _rel_prob(rel_probs, e1, e2)
        card = str(rel.get('cardinality', '')).upper()

        if card in ('N:M', 'M:N'):
            # Resolve associative entity name
            assoc_info = rel.get('associative_entity')
            if assoc_info and isinstance(assoc_info, dict) and assoc_info.get('name'):
                assoc_name  = str(assoc_info['name']).strip().upper()
                assoc_attrs = assoc_info.get('attributes', [])
            else:
                assoc_name  = f'ASSOC_{e1}_{e2}'.upper()
                assoc_attrs = []

            # Add / merge associative entity
            if assoc_name not in out_entities:
                out_entities[assoc_name] = prob
            else:
                out_entities[assoc_name] = max(out_entities[assoc_name], prob)

            if assoc_name not in out_attributes:
                out_attributes[assoc_name] = {}

            # Primary key + foreign keys of associative entity
            for key_name in [f'{assoc_name.lower()}_id',
                              f'{e1.lower()}_id',
                              f'{e2.lower()}_id'] + assoc_attrs:
                prev = out_attributes[assoc_name].get(key_name, 0.0)
                out_attributes[assoc_name][key_name] = max(prev, prob)

            # Expand N:M → two 1:N relationships
            for new_e1, new_e2 in [(e1, assoc_name), (e2, assoc_name)]:
                new_rel = {
                    'entity_1':         new_e1,
                    'entity_2':         new_e2,
                    'description':      f'One {new_e1} can be associated with many {assoc_name}.',
                    'cardinality':      '1:N',
                    'associative_entity': None,
                    'probability':      prob,
                }
                # Merge if duplicate
                merged = False
                for r in out_relationships:
                    if r['entity_1'] == new_e1 and r['entity_2'] == new_e2:
                        r['probability'] = max(r['probability'], prob)
                        merged = True
                        break
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
    out.pop('attribut', None)   # clean up alternate key if present
    return out


# ── Process a single JSON file ────────────────────────────────────────────────
def process_file(args: tuple) -> bool:
    """
    args = (filename, gen_folder, out_folder)
    Returns True on success, False on failure.
    """
    filename, gen_folder, out_folder = args
    try:
        t0 = time.time()

        # Resolve input text file (dataset uses plain numbers, e.g. 251.txt)
        txt_name = filename.replace('.json', '.txt')
        txt_path = os.path.join(TXT_ROOT, txt_name)
        if not os.path.exists(txt_path):
            print(f'  [SKIP] No text file for {filename}')
            return False

        json_path = os.path.join(gen_folder, filename)
        with open(txt_path,  'r', encoding='utf-8') as f:
            raw_text = f.read().strip()
        with open(json_path, 'r', encoding='utf-8') as f:
            gen_data = json.load(f)

        processed_text = preprocess_cached(raw_text)

        entities      = gen_data.get('entity', [])
        attributes    = gen_data.get('attribute', gen_data.get('attribut', {}))
        relationships = gen_data.get('relationship', [])

        # 1. Entity probabilities (text-based, unchanged from original)
        entity_conf  = entity_similarity_checker(processed_text, entities)
        entity_probs = extract_entity_probs(
            compute_entity_probabilities(entity_conf, bias=BIAS, weight=WEIGHT)
        )

        # 2. Attribute probabilities (ablation: x_text only, no Wikidata)
        attribute_probs = extract_attribute_probs(
            compute_all_attribute_probabilities(
                attributes, bias=BIAS, weight=WEIGHT,
                processed_text=processed_text,
            )
        )

        # 3. Relationship probabilities (ablation: 3 features, ablation MLP)
        rel_probs = compute_relationship_probabilities_2(
            json.dumps(relationships), processed_text,
            bias=BIAS, weight=WEIGHT,
        )

        out_data = build_output(
            gen_data, entities, attributes, relationships,
            entity_probs, attribute_probs, rel_probs,
        )
        out_data['processing_time'] = round(time.time() - t0, 2)

        out_path = os.path.join(out_folder, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

        print(f'  [OK] {filename}  ({out_data["processing_time"]:.1f}s)')
        return True

    except Exception as e:
        print(f'  [ERR] {filename}: {e}')
        return False


# ── Setup per-variant logger ──────────────────────────────────────────────────
def _setup_logger(variant_key: str) -> logging.Logger:
    """
    Create a logger that writes to both stdout and
    log/{variant_key}.txt (variant_key uses '/' → '-' for filename).
    """
    os.makedirs(LOG_ROOT, exist_ok=True)
    log_name = variant_key.replace(os.sep, '-').replace('/', '-')
    log_path = os.path.join(LOG_ROOT, f'{log_name}.txt')

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()          # avoid duplicate handlers on re-run

    fmt = logging.Formatter('%(message)s')

    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    return logger, log_path


# ── Run one variant ───────────────────────────────────────────────────────────
def run_variant(gen_sub: str, out_sub: str, max_workers: int):
    gen_folder = os.path.join(GEN_ROOT, gen_sub)
    out_folder = os.path.join(OUT_ROOT, out_sub)

    logger, log_path = _setup_logger(out_sub)

    if not os.path.isdir(gen_folder):
        logger.info(f'  [WARN] Generation folder not found: {gen_folder}')
        return 0, 0

    os.makedirs(out_folder, exist_ok=True)

    json_files = sorted(f for f in os.listdir(gen_folder) if f.endswith('.json'))
    logger.info(f'\n{"="*65}')
    logger.info(f'Variant : {gen_sub}')
    logger.info(f'Files   : {len(json_files)}')
    logger.info(f'Output  : {out_folder}')
    logger.info(f'Log     : {log_path}')
    logger.info(f'{"="*65}')

    t0   = time.time()
    args = [(fn, gen_folder, out_folder) for fn in json_files]

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for result in pool.map(process_file, args):
            if result: ok   += 1
            else:      fail += 1

    elapsed = time.time() - t0
    logger.info(f'\n  Done: {ok} OK, {fail} failed, {elapsed:.1f}s')
    logger.info(f'  Log saved → {log_path}')
    return ok, fail


# ── CLI entry point ───────────────────────────────────────────────────────────
def main():
    # Declare globals first — must precede any use of these names in the function
    global BIAS, WEIGHT, OUT_ROOT

    parser = argparse.ArgumentParser(
        description='Ablation probability estimation for all generation variants.'
    )
    parser.add_argument(
        '--variant', default=None,
        help='Run only the variant whose gen_sub contains this string (e.g. "few-shot-gpt").'
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help='ThreadPoolExecutor workers per variant (default: 8).'
    )
    parser.add_argument(
        '--bias', type=float, default=BIAS,
        help=f'Sigmoid bias for entity/attribute calibration (default: {BIAS}).'
    )
    parser.add_argument(
        '--weight', type=float, default=WEIGHT,
        help=f'Sigmoid weight for entity/attribute calibration (default: {WEIGHT}).'
    )
    args = parser.parse_args()

    BIAS   = args.bias
    WEIGHT = args.weight

    # Encode bias/weight in output root so multiple runs don't overwrite each other
    # e.g. output_0.5_1.0/  or  output_-0.5_1.0/
    OUT_ROOT = os.path.join(_SELF_DIR, f'output_{BIAS}_{WEIGHT}')

    variants = VARIANTS
    if args.variant:
        variants = [(g, o) for g, o in VARIANTS if args.variant in g]
        if not variants:
            print(f'No variant matches "{args.variant}". Available:')
            for g, _ in VARIANTS:
                print(f'  {g}')
            sys.exit(1)

    print(f'bias={BIAS}  weight={WEIGHT}  output_root={OUT_ROOT}')

    total_ok = total_fail = 0
    for gen_sub, out_sub in variants:
        ok, fail = run_variant(gen_sub, out_sub, args.workers)
        total_ok   += ok
        total_fail += fail

    print(f'\n{"="*65}')
    print(f'All variants done.  OK={total_ok}  Failed={total_fail}')
    print(f'Output root: {OUT_ROOT}')
    print(f'{"="*65}')


if __name__ == '__main__':
    main()
