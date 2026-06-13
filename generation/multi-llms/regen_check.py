"""
Regenerate specific exercises that previously had no relationships.
Target IDs: 363, 408, 410, 411, 413, 423, 424, 428, 451, 469
Output: output/generation/multi-llms/few-shot-llama/
"""

import os
import json
import time
import sys
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pre_processing import preprocess_text
from src.entity_processing import (
    entity_few_shot_prompting_without_description_1,
    extract_llama_entities_from_text
)
from src.attribute_processing import (
    attribute_few_shot_prompting,
    extract_attribute_names,
    normalize_schema
)
from src.relationship_processing import (
    relationship_few_shot_prompting,
    extract_relationships
)

# ── Config ────────────────────────────────────────────────────────────────────

TARGET_IDS   = [363, 408, 410, 411, 413, 423, 424, 428, 451, 469]
INPUT_FOLDER = "dataset/Datasets/Full-Dataset/input"
OUTPUT_FOLDER = "output/generation/multi-llms/few-shot-llama"
COST_OUTPUT_FILE = "output/cost_regen_no_rel_llama.txt"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TOGETHER_API_KEY = "put your api key here"
TOGETHER_MODEL   = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

client = OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/v1",
)

total_prompt_tokens = 0
total_completion_tokens = 0
total_calls = 0
token_usage_by_file = {}
batch_total_tokens = 0
batch_total_cost = 0
batch_files_processed = 0
stats_lock = threading.Lock()


def extract_json_from_text(text: str) -> dict:
    if not text: return {}
    try: return json.loads(text)
    except: pass
    try:
        s = text.find("{")
        e = text.rfind("}") + 1
        if s != -1 and e > s: return json.loads(re.sub(r'\n\s+', ' ', text[s:e]))
    except: pass
    return {}


def safe_extract_attribute_names(text: str) -> dict:
    if not text: return {}
    try:
        res = extract_attribute_names(text)
        if res and isinstance(res, dict): return res
    except: pass
    res = extract_json_from_text(text)
    if res and isinstance(res, dict):
        if "attributes" in res: return res["attributes"]
        return res
    return {}


def call_llm(task_name, system_content, user_prompt, max_tokens=2048):
    global total_prompt_tokens, total_completion_tokens, total_calls
    try:
        response = client.chat.completions.create(
            model=TOGETHER_MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        response_content = response.choices[0].message.content or ""
        with stats_lock:
            total_calls += 1
            if hasattr(response, "usage") and response.usage:
                total_prompt_tokens     += response.usage.prompt_tokens or 0
                total_completion_tokens += response.usage.completion_tokens or 0
            else:
                total_prompt_tokens     += len(user_prompt) // 4
                total_completion_tokens += len(response_content) // 4
        return response_content
    except Exception as e:
        print(f"[{task_name}] LLM Error: {e}")
        return ""


def process_file(filename):
    global batch_total_tokens, batch_total_cost, batch_files_processed

    output_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(filename)[0] + ".json")

    # Skip if already regenerated
    if os.path.exists(output_path):
        return True

    file_path = os.path.join(INPUT_FOLDER, filename)
    start_time = time.time()

    with stats_lock:
        file_start_prompt     = total_prompt_tokens
        file_start_completion = total_completion_tokens
        file_start_calls      = total_calls

    try:
        print(f"[START] {filename} ...")
        with open(file_path, encoding="utf-8") as f:
            raw_description = f.read().strip()

        if not raw_description:
            print(f"  [SKIP] {filename} is empty.")
            return False

        # 1. Entities
        entity_prompt = entity_few_shot_prompting_without_description_1(raw_description)
        entity_system = (
            "You are a database design expert specializing in Entity-Relationship (ER) modeling. "
            "Your task is to identify the minimal set of core entities needed to represent the described system. "
            "Be conservative: only extract entities that are clearly distinct objects participating in relationships."
        )
        generated_entity = call_llm("entity", entity_system, entity_prompt)
        entities_list = extract_llama_entities_from_text(generated_entity)
        if not entities_list:
            print(f"  [WARN] No entities for {filename}. Skipping.")
            return False

        # 2. Attributes
        attribute_prompt = attribute_few_shot_prompting(raw_description, entities_list)
        attribute_system = (
            "You are a database design expert. "
            "For each entity in the ER schema, extract only the attributes explicitly mentioned in the description "
            "plus a primary key if not stated. Do not add speculative or redundant attributes."
        )
        attribute_llm = call_llm("attribute", attribute_system, attribute_prompt)
        attrs_raw = safe_extract_attribute_names(attribute_llm)
        normalized_attrs = normalize_schema(attrs_raw)

        # 3. Relationships — prompt emphasizes not to omit relationships
        relationship_prompt_text = relationship_few_shot_prompting(raw_description, normalized_attrs)
        relationship_system = (
            "You are a database design expert specializing in ER modeling. "
            "Identify ALL semantic relationships between entities based on the description. "
            "Do NOT omit relationships — if two entities interact in the description, define a relationship. "
            "Determine correct cardinalities (1:1, 1:N, N:M). "
            "For each entity pair, define AT MOST ONE relationship."
        )
        relationship_llm = call_llm("relationship", relationship_system, relationship_prompt_text)

        try:
            raw_relationships = extract_relationships(relationship_llm)
        except Exception:
            json_data = extract_json_from_text(relationship_llm)
            raw_relationships = json_data.get("relationships", []) if isinstance(json_data, dict) else []

        end_time = time.time()
        processing_time = round(end_time - start_time, 2)

        result_dict = {
            "filename":        filename,
            "entity":          entities_list,
            "attribute":       normalized_attrs,
            "relationship":    raw_relationships,
            "processing_time": processing_time,
        }

        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(result_dict, f_out, ensure_ascii=False, indent=2)

        with stats_lock:
            file_prompt_tokens     = total_prompt_tokens     - file_start_prompt
            file_completion_tokens = total_completion_tokens - file_start_completion
            file_api_calls         = total_calls             - file_start_calls
            file_tokens = file_prompt_tokens + file_completion_tokens
            input_cost  = (file_prompt_tokens     / 1_000_000) * 0.80
            output_cost = (file_completion_tokens / 1_000_000) * 0.90
            file_cost   = input_cost + output_cost
            batch_total_tokens    += file_tokens
            batch_total_cost      += file_cost
            batch_files_processed += 1
            token_usage_by_file[filename] = {
                "total_tokens":     file_tokens,
                "estimated_cost":   file_cost,
                "api_calls":        file_api_calls,
                "processing_time":  processing_time,
            }

        print(
            f"[SUCCESS] {filename} | "
            f"E:{len(entities_list)} A:{sum(len(v) for v in normalized_attrs.values())} "
            f"R:{len(raw_relationships)} | Time:{processing_time}s"
        )
        return True

    except Exception as e:
        print(f"[ERROR] {filename}: {e}")
        return False


if __name__ == "__main__":
    # Build filenames for target IDs
    input_files = []
    for ex_id in TARGET_IDS:
        fname = f"{ex_id}.txt"
        fpath = os.path.join(INPUT_FOLDER, fname)
        if os.path.exists(fpath):
            input_files.append(fname)
        else:
            print(f"[WARN] Input not found: {fpath}")

    print(f"\nRegenerating {len(input_files)} exercises: {TARGET_IDS}")
    print(f"Output → {OUTPUT_FOLDER}")
    print("=" * 60)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_file, f): f for f in input_files}
        for future in as_completed(futures):
            future.result()

    # Save cost log
    lines = [
        f"Regenerated exercises: {TARGET_IDS}",
        f"Files processed: {batch_files_processed}",
        f"Total tokens: {batch_total_tokens}",
        f"Estimated cost: ${batch_total_cost:.4f}",
        "",
    ]
    for fname, info in token_usage_by_file.items():
        lines.append(
            f"  {fname}: tokens={info['total_tokens']} "
            f"cost=${info['estimated_cost']:.4f} calls={info['api_calls']} "
            f"time={info['processing_time']}s  R=?"
        )

    with open(COST_OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines))

    print(f"\nDone. Cost log → {COST_OUTPUT_FILE}")
