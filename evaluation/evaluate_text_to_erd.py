import os
import json
import re
import csv
import numpy as np
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util
from scipy.optimize import linear_sum_assignment
import spacy

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GENERATION_DIR = os.path.join(project_root, "output/generation/Text-To-ERD/llama")
REF_DIR = os.path.join(project_root, "dataset/Datasets/Full-Dataset/Reference")
if not os.path.exists(REF_DIR):
    REF_DIR = os.path.join(project_root, "dataset/Datasets/Reference")

RESULTS_DIR = os.path.join(project_root, "results/F1Score/Text-To-ERD")

# ==========================================
# 2. MODEL INITIALIZATION
# ==========================================
print("Initializing BERT model (SentenceTransformer)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Initializing spaCy...")
spacy_nlp = spacy.load("en_core_web_sm")
spacy_nlp.Defaults.stop_words.add("record")

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================

def clean_name(s):
    if not s:
        return ""
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return s.replace('_', ' ').replace('-', ' ').lower().strip()


_NOISE_SUFFIXES = {"table", "record", "information", "data", "metadata",
                   "file", "sheet", "db", "database", "list", "detail", "details"}

def strip_noise_suffix(name: str) -> str:
    words = clean_name(name).split()
    while len(words) > 1 and words[-1] in _NOISE_SUFFIXES:
        words.pop()
    return " ".join(words)


def are_synonyms(word1, word2):
    if word1.lower() == word2.lower():
        return True
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)
    for s1 in synsets1:
        for s2 in synsets2:
            if s1 == s2:
                return True
    return False


def are_synonyms_phrase(phrase1, phrase2):
    def normalize(phrase):
        s = clean_name(phrase)
        s = re.sub(r'\bnumber\b', 'id', s)
        return [w for w in s.split() if w]
    p1 = normalize(phrase1)
    p2 = normalize(phrase2)
    if not p1 or not p2 or len(p1) != len(p2):
        return False
    return all(are_synonyms(w1, w2) for w1, w2 in zip(p1, p2))


def char_lcs_score(s1, s2):
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                best = max(best, dp[i][j])
    return best / min(m, n)


def strict_word_overlap(sent1, sent2, min_shared=2, char_lcs_thre=0.85):
    doc1 = spacy_nlp(sent1)
    doc2 = spacy_nlp(sent2)
    words1 = {t.lemma_ for t in doc1 if not (t.is_stop or t.is_punct)}
    words2 = {t.lemma_ for t in doc2 if not (t.is_stop or t.is_punct)}
    if not words1 or not words2:
        return False
    if len(words1) == 1 and len(words2) == 1:
        w1, w2 = next(iter(words1)), next(iter(words2))
        return char_lcs_score(w1, w2) >= 1.0
    exact_shared = len(words1 & words2)
    near_match = sum(
        1 for w1 in words1 - words2
        for w2 in words2 - words1
        if char_lcs_score(w1, w2) >= char_lcs_thre
    )
    return (exact_shared + near_match) >= min_shared


def get_smart_mapping_v2(list_out, list_ref, bert_threshold=0.50):
    if not list_out or not list_ref:
        return {}
    mapping = {}
    used_ref = set()

    def remaining_out():
        return [x for x in list_out if x not in mapping]

    def remaining_ref():
        return [x for x in list_ref if x not in used_ref]

    for out in list_out:
        for ref in list_ref:
            if ref not in used_ref and strip_noise_suffix(out) == strip_noise_suffix(ref):
                mapping[out] = ref
                used_ref.add(ref)
                break

    for out in remaining_out():
        for ref in remaining_ref():
            if clean_name(out) == clean_name(ref):
                mapping[out] = ref
                used_ref.add(ref)
                break

    for out in remaining_out():
        for ref in remaining_ref():
            if are_synonyms_phrase(out, ref):
                mapping[out] = ref
                used_ref.add(ref)
                break

    rem_out = remaining_out()
    rem_ref = remaining_ref()
    if rem_out and rem_ref:
        cleaned_out = [strip_noise_suffix(i) for i in rem_out]
        cleaned_ref = [strip_noise_suffix(i) for i in rem_ref]
        emb_out = model.encode(cleaned_out, convert_to_tensor=True)
        emb_ref = model.encode(cleaned_ref, convert_to_tensor=True)
        cosine_matrix = util.cos_sim(emb_out, emb_ref).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(1 - cosine_matrix)
        for r, c in zip(row_ind, col_ind):
            if cosine_matrix[r, c] >= bert_threshold:
                out_item = rem_out[r]
                ref_item = rem_ref[c]
                if out_item not in mapping and ref_item not in used_ref:
                    mapping[out_item] = ref_item
                    used_ref.add(ref_item)

    for out in remaining_out():
        for ref in remaining_ref():
            if strict_word_overlap(clean_name(out), clean_name(ref)):
                mapping[out] = ref
                used_ref.add(ref)
                break

    return mapping


def calc_metrics(tp, total_out, total_ref):
    p = tp / total_out if total_out > 0 else 0
    r = tp / total_ref if total_ref > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1


# ==========================================
# 4. EVALUATE A SINGLE EXERCISE
# ==========================================

def process_single_exercise(out_path, ref_path):
    with open(out_path, 'r', encoding='utf-8') as f:
        out_data = json.load(f)
    with open(ref_path, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)

    out_entities_all = out_data.get("entity", [])
    out_entities_main = [e for e in out_entities_all if not str(e).upper().startswith("ASSOC_")]
    ref_entities = ref_data.get("entity", [])

    def evaluate_with_entities(entities_to_use):
        e_map = get_smart_mapping_v2(entities_to_use, ref_entities)
        p_e, r_e, f1_e = calc_metrics(len(e_map), len(entities_to_use), len(ref_entities))
        entity_all_correct = 1 if abs(f1_e - 1.0) < 1e-6 else 0

        tp_a, total_oa, total_ra = 0, 0, 0
        attr_all_correct_sum = 0
        for out_ent, ref_ent in e_map.items():
            oa = out_data.get("attribut", out_data.get("attribute", {})).get(out_ent, [])
            ra = ref_data.get("attribut", ref_data.get("attribute", {})).get(ref_ent, [])
            a_map = get_smart_mapping_v2(oa, ra)
            tp_a += len(a_map)
            total_oa += len(oa)
            total_ra += len(ra)
            _, _, attr_f1_single = calc_metrics(len(a_map), len(oa), len(ra))
            attr_all_correct_sum += 1 if abs(attr_f1_single - 1.0) < 1e-6 else 0

        p_a, r_a, f1_a = calc_metrics(tp_a, total_oa, total_ra)
        attribute_all_correct = (attr_all_correct_sum / len(e_map)) if e_map else 0.0

        def get_edges(data, valid_entities=None, mapping=None):
            edges = set()
            for r in data.get("relationship", []):
                e1, e2 = r.get("entity_1"), r.get("entity_2")
                if valid_entities is not None:
                    if e1 not in valid_entities or e2 not in valid_entities:
                        continue
                if mapping:
                    e1, e2 = mapping.get(e1), mapping.get(e2)
                if e1 and e2:
                    edges.add(tuple(sorted((str(e1), str(e2)))))
            return edges

        out_rel = get_edges(out_data, valid_entities=set(entities_to_use), mapping=e_map)
        ref_rel = get_edges(ref_data)

        e_map_keys = set(e_map.keys())
        total_out_rel = sum(
            1 for r in out_data.get("relationship", [])
            if r.get("entity_1") in e_map_keys and r.get("entity_2") in e_map_keys
        )
        total_ref_rel = len(ref_rel)
        tp_r = len(out_rel & ref_rel)
        p_r, r_r, f1_r = calc_metrics(tp_r, total_out_rel, total_ref_rel)
        relation_all_correct = 1 if abs(f1_r - 1.0) < 1e-6 else 0

        full_all_correct = 1 if (entity_all_correct == 1 and attribute_all_correct == 1.0 and relation_all_correct == 1) else 0

        return {
            "entity":                (p_e, r_e, f1_e),
            "attribute":             (p_a, r_a, f1_a),
            "relation":              (p_r, r_r, f1_r),
            "entity_all_correct":    entity_all_correct,
            "attribute_all_correct": attribute_all_correct,
            "relation_all_correct":  relation_all_correct,
            "full_all_correct":      full_all_correct,
            "overall_f1":            (f1_e + f1_a + f1_r) / 3,
        }

    res_main = evaluate_with_entities(out_entities_main)
    res_all = evaluate_with_entities(out_entities_all)
    return res_main if res_main["overall_f1"] >= res_all["overall_f1"] else res_all


# ==========================================
# 5. RUN EVALUATION
# ==========================================
os.makedirs(RESULTS_DIR, exist_ok=True)

csv_result_path = os.path.join(RESULTS_DIR, "text_to_erd_llama.csv")

print("\n" + "=" * 80)
print(f"Starting evaluation: Text-To-ERD / LLaMA")
print(f"Input : {GENERATION_DIR}")
print(f"Output: {csv_result_path}")
print("-" * 80)

csv_rows, summary_list = [], []

for i in range(251, 501):
    out_file = os.path.join(GENERATION_DIR, f"{i}.json")
    ref_file = os.path.join(REF_DIR, f"exercise{i}-baseline.txt")

    if os.path.exists(out_file) and os.path.exists(ref_file):
        try:
            m = process_single_exercise(out_file, ref_file)
            summary_list.append(m)
            print(
                f"Ex {i:<4} | Ent F1: {m['entity'][2]:.2f} | "
                f"Attr F1: {m['attribute'][2]:.2f} | "
                f"Rel F1: {m['relation'][2]:.2f} | "
                f"AllCorrect(E/A/R/Full): "
                f"{m['entity_all_correct']}/{m['attribute_all_correct']:.2f}/"
                f"{m['relation_all_correct']}/{m['full_all_correct']}"
            )
            csv_rows.append([
                f"Ex {i}",
                *[f"{v:.4f}" for v in m['entity']],
                *[f"{v:.4f}" for v in m['attribute']],
                *[f"{v:.4f}" for v in m['relation']],
                m['entity_all_correct'],
                f"{m['attribute_all_correct']:.4f}",
                m['relation_all_correct'],
                m['full_all_correct'],
            ])
        except Exception as e:
            print(f"Ex {i:<4} | ERROR: {e}")

if summary_list:
    with open(csv_result_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Exercise",
            "Ent_P", "Ent_R", "Ent_F1",
            "Attr_P", "Attr_R", "Attr_F1",
            "Rel_P", "Rel_R", "Rel_F1",
            "Ent_AllCorrect", "Attr_AllCorrect", "Rel_AllCorrect", "Full_AllCorrect",
        ])
        writer.writerows(csv_rows)
        avg_row = ["AVERAGE"] + [
            round(np.mean([x[k][j] for x in summary_list]), 4)
            for k in ['entity', 'attribute', 'relation'] for j in range(3)
        ] + [
            round(np.mean([x['entity_all_correct'] for x in summary_list]), 4),
            round(np.mean([x['attribute_all_correct'] for x in summary_list]), 4),
            round(np.mean([x['relation_all_correct'] for x in summary_list]), 4),
            round(np.mean([x['full_all_correct'] for x in summary_list]), 4),
        ]
        writer.writerow(avg_row)
    print(
        f"\nAVERAGE -> Ent F1: {avg_row[3]:.4f} | "
        f"Attr F1: {avg_row[6]:.4f} | "
        f"Rel F1: {avg_row[9]:.4f} | "
        f"AllCorrect(E/A/R/Full): {avg_row[10]}/{avg_row[11]}/{avg_row[12]}/{avg_row[13]}"
    )
    print(f"\nResults saved to: {csv_result_path}")
else:
    print("No files found for evaluation.")
