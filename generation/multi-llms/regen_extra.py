"""
Regenerate 18 failed files (attribute=0 or relationship=0) using the Groq API.
Output: output/add/
"""
import os, json, time, sys, re, threading
from itertools import cycle
from openai import OpenAI

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.entity_processing import (
    entity_few_shot_prompting_without_description_1,
    extract_llama_entities_from_text,
)
from src.attribute_processing import (
    attribute_few_shot_prompting,
    extract_attribute_names,
    normalize_schema,
)
from src.relationship_processing import (
    relationship_few_shot_prompting,
    extract_relationships,
)

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FOLDER  = os.path.join(project_root, "dataset/Datasets/Full-Dataset/input")
OUTPUT_FOLDER = os.path.join(project_root, "output/add")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# 18 files to regenerate
# 17 files with attribute=0 + 1 file with relationship=0
TARGET_FILES = [
    "319.txt", "344.txt", "363.txt", "365.txt",
    "418.txt", "422.txt", "427.txt", "431.txt",
    "432.txt", "434.txt", "441.txt", "455.txt",
    "471.txt", "472.txt", "476.txt", "480.txt",
    "491.txt", "494.txt",
]

# Files to retry (attribute=0 after first run)
RETRY_FILES = ["472.txt", "476.txt"]

# Groq API keys from config.py
GROQ_API_KEYS = [
    "put yoru api key here"
]

# Key rotation state
_key_index = 0
_key_lock  = threading.Lock()

def get_next_key():
    global _key_index
    with _key_lock:
        key = GROQ_API_KEYS[_key_index % len(GROQ_API_KEYS)]
        _key_index += 1
        return key

def make_client(api_key):
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


# ── LLM call with key rotation on rate limit ─────────────────────────────────
def call_llm(task_name, system_content, user_prompt, max_retries=5):
    for attempt in range(max_retries):
        key = get_next_key()
        client = make_client(key)
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=8192,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 10 * (attempt + 1)
                print(f"  [{task_name}] Rate limit (key …{key[-6:]}), waiting {wait}s …")
                time.sleep(wait)
            else:
                print(f"  [{task_name}] Error (attempt {attempt+1}): {err[:80]}")
                time.sleep(3)
    print(f"  [{task_name}] All retries exhausted, returning empty string")
    return ""


# ── Fallback JSON extractor (same as generation script) ──────────────────────
def extract_json_from_text(text):
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        s = text.find("{")
        e = text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(re.sub(r'\n\s+', ' ', text[s:e]))
    except Exception:
        pass
    return {}


def _extract_attr_name_from_dict(attr_dict):
    """Handle all known LLM dict formats for an attribute object."""
    for key in ("name", "attribute_name", "column_name", "attribute", "field_name", "attr_name"):
        v = attr_dict.get(key)
        if v and isinstance(v, str):
            return v.strip()
    return None


def _parse_attr_list(attrs):
    """Convert a list of attrs (str or dict) to a list of string names."""
    names = []
    for attr in attrs:
        if isinstance(attr, str) and attr.strip():
            names.append(attr.strip())
        elif isinstance(attr, dict):
            name = _extract_attr_name_from_dict(attr)
            if name:
                names.append(name)
    return names


def safe_extract_attribute_names(text):
    if not text:
        return {}
    # Primary: use the standard parser
    try:
        res = extract_attribute_names(text)
        # Only accept if at least one entity actually has attributes
        if res and isinstance(res, dict) and any(v for v in res.values()):
            return res
    except Exception:
        pass
    # Fallback: extract JSON block and parse manually
    res = extract_json_from_text(text)
    if res and isinstance(res, dict):
        src = res.get("attributes", res)
        if isinstance(src, dict):
            return {ent: _parse_attr_list(attrs)
                    for ent, attrs in src.items()
                    if isinstance(attrs, list)}
    return {}


# ── Process one file ──────────────────────────────────────────────────────────
def process_file(txt_filename):
    out_json = os.path.splitext(txt_filename)[0] + ".json"
    out_path = os.path.join(OUTPUT_FOLDER, out_json)

    if os.path.exists(out_path):
        print(f"[SKIP] {txt_filename} already exists in output")
        return True

    txt_path = os.path.join(INPUT_FOLDER, txt_filename)
    if not os.path.exists(txt_path):
        print(f"[SKIP] {txt_filename} — input txt not found")
        return False

    start = time.time()
    print(f"\n[START] {txt_filename}")

    with open(txt_path, encoding="utf-8") as f:
        raw_description = f.read().strip()

    if not raw_description:
        print(f"  [SKIP] Empty input text")
        return False

    # ── Step 1: Entity extraction ─────────────────────────────────────────────
    entity_prompt = entity_few_shot_prompting_without_description_1(raw_description)
    entity_content = (
        "You are a database design expert specializing in Entity-Relationship (ER) modeling. "
        "Your task is to identify the minimal set of core entities needed to represent the described system. "
        "Be conservative: only extract entities that are clearly distinct objects participating in relationships."
    )
    entity_raw = call_llm("entity", entity_content, entity_prompt)
    entities_list = extract_llama_entities_from_text(entity_raw)

    if not entities_list:
        print(f"  [FAIL] No entities extracted for {txt_filename}")
        return False
    print(f"  Entities ({len(entities_list)}): {entities_list}")

    # ── Step 2: Attribute extraction ──────────────────────────────────────────
    attribute_prompt = attribute_few_shot_prompting(raw_description, entities_list)
    attribute_content = (
        "You are a database design expert. "
        "For each entity in the ER schema, extract only the attributes explicitly mentioned in the description "
        "plus a primary key if not stated. Do not add speculative or redundant attributes."
    )
    attribute_raw = call_llm("attribute", attribute_content, attribute_prompt)
    attrs_raw = safe_extract_attribute_names(attribute_raw)
    normalized_attrs = normalize_schema(attrs_raw)
    total_attrs = sum(len(v) for v in normalized_attrs.values())
    print(f"  Attributes ({total_attrs} total across {len(normalized_attrs)} entities)")

    # ── Step 3: Relationship extraction ──────────────────────────────────────
    relationship_prompt = relationship_few_shot_prompting(raw_description, normalized_attrs)
    relationship_content = (
        "You are a database design expert specializing in ER modeling. "
        "Identify semantic relationships between entities based strictly on the description. "
        "Determine correct cardinalities (1:1, 1:N, N:M). "
        "For each entity pair, define AT MOST ONE relationship — do not create both A→B and B→A."
    )
    relationship_raw = call_llm("relationship", relationship_content, relationship_prompt)

    try:
        raw_relationships = extract_relationships(relationship_raw)
    except Exception:
        json_data = extract_json_from_text(relationship_raw)
        raw_relationships = json_data.get("relationships", []) if isinstance(json_data, dict) else []

    print(f"  Relationships: {len(raw_relationships)}")

    processing_time = round(time.time() - start, 2)

    result = {
        "filename": txt_filename,
        "entity": entities_list,
        "attribute": normalized_attrs,
        "relationship": raw_relationships,
        "processing_time": processing_time,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    a_ok = total_attrs > 0
    r_ok = len(raw_relationships) > 0
    status = "OK" if (a_ok and r_ok) else f"WARN(attr={'OK' if a_ok else '0'} rel={'OK' if r_ok else '0'})"
    print(f"  [{status}] {txt_filename} — {processing_time}s")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    # If --retry flag, only process RETRY_FILES
    files_to_run = RETRY_FILES if "--retry" in sys.argv else TARGET_FILES

    print("=" * 70)
    print(f"Re-generating {len(files_to_run)} files using Groq API")
    print(f"Model : {GROQ_MODEL}")
    print(f"Output: {OUTPUT_FOLDER}")
    print("=" * 70)

    success, failed = 0, []
    for fname in files_to_run:
        ok = process_file(fname)
        if ok:
            success += 1
        else:
            failed.append(fname)
        time.sleep(1)  # brief pause between files

    print("\n" + "=" * 70)
    print(f"Done: {success}/{len(TARGET_FILES)} succeeded")
    if failed:
        print(f"Failed: {failed}")

    # Verify output
    print("\n── Verification ──")
    for fname in TARGET_FILES:
        out_json = os.path.splitext(fname)[0] + ".json"
        out_path = os.path.join(OUTPUT_FOLDER, out_json)
        if os.path.exists(out_path):
            with open(out_path) as f:
                d = json.load(f)
            total_a = sum(len(v) for v in d.get("attribute", {}).values())
            total_r = len(d.get("relationship", []))
            flag = "" if (total_a > 0 and total_r > 0) else " ← STILL EMPTY"
            print(f"  {out_json}: E={len(d.get('entity',[]))} A={total_a} R={total_r}{flag}")
        else:
            print(f"  {out_json}: NOT GENERATED")
