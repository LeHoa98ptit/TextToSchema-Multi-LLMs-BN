import os
import json
import time
import sys
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root directory to sys.path so Python can find the 'src' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import GROQ_API_KEYS, GPT_MODEL
from src.pre_processing import preprocess_text
from src.setup_llm import MultiKeyGroqManager

from src.entity_processing import (
    entity_prompting_without_description_1, 
    extract_gpt_entities
)
from src.attribute_processing import (
    attribute_prompting,
    extract_attribute_names,
    normalize_schema
)
from src.relationship_processing import (
    relationship_prompting,
    extract_relationships
)

# -------------------------
# Configuration & Paths
# -------------------------
# Customize the range of files to process here
START_FILE_INDEX = 261
END_FILE_INDEX = 500

INPUT_FOLDER = "dataset/Datasets/Full-Dataset/input"
OUTPUT_FOLDER = "output/generation/multi-llms/zero-shot-gpt"
COST_OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "cost_report_zeroshot_gpt.txt")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------
# Setup LLM Client & Token Tracking
# -------------------------
client = MultiKeyGroqManager(api_keys=GROQ_API_KEYS)
gpt_model = GPT_MODEL

# Variables for token tracking
total_prompt_tokens = 0
total_completion_tokens = 0
total_calls = 0
token_usage_by_file = {}
batch_total_tokens = 0
batch_total_cost = 0
batch_files_processed = 0

# Lock for thread-safe token counter updates
stats_lock = threading.Lock()

# -------------------------
# Improved JSON extraction
# -------------------------
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

# -------------------------
# Call LLM with Token Tracking
# -------------------------
def call_llm_with_logging_tracking(task_name, system_content, client, model, user_prompt, temperature=0.3, max_tokens=7000):
    global total_prompt_tokens, total_completion_tokens, total_calls
    
    try:
        with stats_lock:
            total_calls += 1
        
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # Call API
        if hasattr(client, 'create_chat_completion'):
            response = client.create_chat_completion(**kwargs)
        else:
            response = client.chat.completions.create(**kwargs)
        
        response_content = response.choices[0].message.content
        
        # Track token usage
        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            with stats_lock:
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
        else:
            estimated_prompt_tokens = len(user_prompt) // 4
            estimated_completion_tokens = len(response_content) // 4
            with stats_lock:
                total_prompt_tokens += estimated_prompt_tokens
                total_completion_tokens += estimated_completion_tokens
        
        return response_content
        
    except Exception as e:
        print(f"[{task_name}] LLM Call Error: {e}")
        return ""

# -------------------------
# Core Processing Function
# -------------------------
def process_single_file(filename):
    global batch_total_tokens, batch_total_cost, batch_files_processed
    
    output_filename = os.path.splitext(filename)[0] + ".json"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    if os.path.exists(output_path):
        return True

    file_path = os.path.join(INPUT_FOLDER, filename)
    start_time = time.time()
    
    with stats_lock:
        file_start_prompt = total_prompt_tokens
        file_start_completion = total_completion_tokens
        file_start_calls = total_calls

    try:
        print(f"[START] Processing {filename} ...")

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_description = f.read().strip()

        if not raw_description:
            print(f"Warning: {filename} is empty. Skipping.")
            return False

        processed_data = preprocess_text(raw_description)

        # 1. ENTITY EXTRACTION (Zero-Shot)
        entity_prompt = entity_prompting_without_description_1(processed_data)
        entity_content = "You are a database design expert. Identify the main entities for an ER schema from the description below."
        
        generated_entity = call_llm_with_logging_tracking("entity", entity_content, client, gpt_model, entity_prompt)
        entities_list = extract_gpt_entities(generated_entity)
        
        if not entities_list:
            print(f"Warning: No entities extracted for {filename}. Skipping.")
            return False

        # 2. ATTRIBUTE EXTRACTION (Zero-Shot)
        attribute_prompt = attribute_prompting(processed_data, entities_list)
        attribute_content = "Identify attributes for each entity."
        
        attribute_gpt = call_llm_with_logging_tracking("attribute", attribute_content, client, gpt_model, attribute_prompt)
        attrs_raw = safe_extract_attribute_names(attribute_gpt)
        normalized_attrs = normalize_schema(attrs_raw)

        # 3. RELATIONSHIP EXTRACTION (Zero-Shot)
        relationship_prompt_text = relationship_prompting(processed_data, normalized_attrs)
        relationship_content = "Identify relationships between entities."
        
        relationship_gpt = call_llm_with_logging_tracking("relationship", relationship_content, client, gpt_model, relationship_prompt_text)
        
        try:
            raw_relationships = extract_relationships(relationship_gpt)
        except Exception:
            json_data = extract_json_from_text(relationship_gpt)
            raw_relationships = json_data.get("relationships", []) if isinstance(json_data, dict) else []
            
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)

        # 4. BUILD RAW RESULT & SAVE
        result_dict = {
            "filename": filename,
            "entity": entities_list,
            "attribute": normalized_attrs,
            "relationship": raw_relationships,
            "processing_time": processing_time
        }

        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(result_dict, f_out, ensure_ascii=False, indent=2)

        # 5. CALCULATE TOKEN USAGE
        with stats_lock:
            file_prompt_tokens = total_prompt_tokens - file_start_prompt
            file_completion_tokens = total_completion_tokens - file_start_completion
            file_api_calls = total_calls - file_start_calls
            file_tokens = file_prompt_tokens + file_completion_tokens
            
            input_cost = (file_prompt_tokens / 1_000_000) * 0.50
            output_cost = (file_completion_tokens / 1_000_000) * 1.50
            file_cost = input_cost + output_cost
            
            batch_total_tokens += file_tokens
            batch_total_cost += file_cost
            batch_files_processed += 1
            
            token_usage_by_file[filename] = {
                'input_length_chars': len(raw_description),
                'total_tokens': file_tokens,
                'estimated_cost': file_cost,
                'api_calls': file_api_calls,
                'processing_time': processing_time
            }

        print(f"[SUCCESS] {filename} | E:{len(entities_list)} A:{sum(len(v) for v in normalized_attrs.values())} R:{len(raw_relationships)} | Time: {processing_time}s")
        return True

    except Exception as e:
        print(f"ERROR processing {filename}: {e}")
        return False

# -------------------------
# MULTITHREADING EXECUTION
# -------------------------
if __name__ == "__main__":
    input_files = []
    for f in os.listdir(INPUT_FOLDER):
        if f.lower().endswith('.txt'):
            nums = re.findall(r'\d+', f)
            if nums:
                idx = int(nums[-1])
                if START_FILE_INDEX <= idx <= END_FILE_INDEX:
                    input_files.append(f)
    
    input_files.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
    
    print(f"\nFOUND {len(input_files)} FILES TO PROCESS (FROM INDEX {START_FILE_INDEX} TO {END_FILE_INDEX})...")
    print(f"Output directory: {OUTPUT_FOLDER}")
    print("="*70)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_single_file, filename): filename for filename in input_files}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Error occurred during multithreading: {exc}")

    # Write cost log to text file
    with open(COST_OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"--- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"Total Files Processed: {batch_files_processed}\n")
        f.write(f"Total Tokens Used: {batch_total_tokens}\n")
        f.write(f"Total Estimated Cost: ${batch_total_cost:.6f}\n")
        f.write("="*50 + "\n")
    
    print(f"\nCOMPLETED. Cost log saved at: {COST_OUTPUT_FILE}")