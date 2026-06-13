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

from src.pre_processing import preprocess_text
from src.setup_llm import MultiKeyGroqManager
from src.config import GROQ_API_KEYS, LLAMA_MODEL

# -------------------------
# Configuration & Paths
# -------------------------
# Customize the range of files to process here (e.g., 251 -> 500)
START_FILE_INDEX = 289
END_FILE_INDEX = 500

INPUT_FOLDER = "dataset/Datasets/Full-Dataset/input"
OUTPUT_FOLDER = "output/generation/one-llm/one_llm_few_shot_llama"
COST_OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "cost_report_onestep_fewshot_llama.txt")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------
# Setup LLM Client & Token Tracking
# -------------------------
client = MultiKeyGroqManager(api_keys=GROQ_API_KEYS)
llama = LLAMA_MODEL

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
# Prompt Function
# -------------------------
def er_one_step_few_shot_prompt(processed_text: str) -> str:
    return f"""
From the following domain description, generate a FINAL Entity-Relationship (ER) model.

DOMAIN DESCRIPTION:
\"\"\"
{processed_text}
\"\"\"

STRICT RULES:
- Output ONLY valid JSON
- Use ONLY English
- Entity names MUST be uppercase
- Attribute names MUST be lowercase
- Attributes must be grouped by entity
- Relationships are pairs of entities
- With relationship has cardinality N:M, so you need to process it

JSON FORMAT:
{{
  "entities": ["ENTITY1", "ENTITY2"],
  "attributes": {{
    "ENTITY1": ["attr1", "attr2"],
    "ENTITY2": ["attrA"]
  }},
  "relationships": [
    {{
      "entity_1": "ENTITY1",
      "entity_2": "ENTITY2",
      "cardinality": "1:1|1:N|N:M"
    }}
  ]
}}
Example: 
    This is a dataset description: The dataset contains 50,000 rows.
    Each row corresponds to one medical appointment at a hospital.
    For each appointment, the dataset stores:
    information about the patient (name, date of birth, gender, phone, address),
    information about the doctor (name, specialty, license number),
    the date and time of the appointment,
    the room where the appointment takes place,
    and the diagnosis and notes written by the doctor
    Given entities: 
    {{
        "entities": ["PATIENT", "DOCTOR", "APPOINTMENT", "CLINIC_ROOM"]
    }}
    Generated attributes: 
    "attributes": {{
        "PATIENT": [
        "patient_id",
        "full_name",
        "date_of_birth",
        "gender",
        "phone_number",
        "address"
        ],
        "DOCTOR": [
        "doctor_id",
        "full_name",
        "specialty",
        "license_number",
        "phone_number"
        ],
        "APPOINTMENT": [
        "appointment_id",
        "appointment_date",
        "appointment_time",
        "diagnosis",
        "notes"
        ],
        "CLINIC_ROOM": [
        "room_id",
        "room_number",
        "floor"
        ]
    }}
    
    "relationships": [
        {{
        "entity_1": "PATIENT",
        "entity_2": "APPOINTMENT",
        "description": "Each appointment is scheduled for exactly one patient, and each patient can have many appointments over time.",
        "cardinality": "1:N",
        "is_identifying": false,
        "associative_entity": null
        }},
        {{
        "entity_1": "DOCTOR",
        "entity_2": "APPOINTMENT",
        "description": "Each appointment is handled by exactly one doctor, and each doctor can handle many appointments.",
        "cardinality": "1:N",
        "is_identifying": false,
        "associative_entity": null
        }},
        {{
        "entity_1": "CLINIC_ROOM",
        "entity_2": "APPOINTMENT",
        "description": "Each appointment takes place in exactly one clinic room, and each room can host many appointments at different times.",
        "cardinality": "1:N",
        "is_identifying": false,
        "associative_entity": null
        }}
        ]
"""

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

# -------------------------
# Call LLM with Token Tracking
# -------------------------
def call_llm_with_logging_tracking(task_name, system_content, client, model, user_prompt, temperature=0.3, max_tokens=5000):
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

        # 1. ONE-STEP EXTRACTION
        prompt = er_one_step_few_shot_prompt(processed_data)
        
        llm_output = call_llm_with_logging_tracking(
            task_name="generate_schema_from_text",
            system_content="You are a database design expert.",
            client=client,
            model=llama,
            user_prompt=prompt
        )

        if not llm_output:
            print(f"Warning: No output generated for {filename}. Skipping.")
            return False

        # SAFE JSON PARSE
        er_model_llm = extract_json_from_text(llm_output)

        entities = er_model_llm.get("entities", [])
        attributes = er_model_llm.get("attributes", {})
        relationships = er_model_llm.get("relationships", [])
            
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)

        # 2. BUILD RAW RESULT & SAVE
        result_dict = {
            "filename": filename,
            "entity": entities,
            "attribute": attributes,
            "relationship": relationships,
            "processing_time": processing_time
        }

        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(result_dict, f_out, ensure_ascii=False, indent=2)

        # 3. CALCULATE TOKEN USAGE
        with stats_lock:
            file_prompt_tokens = total_prompt_tokens - file_start_prompt
            file_completion_tokens = total_completion_tokens - file_start_completion
            file_api_calls = total_calls - file_start_calls
            file_tokens = file_prompt_tokens + file_completion_tokens
            
            input_cost = (file_prompt_tokens / 1_000_000) * 0.80
            output_cost = (file_completion_tokens / 1_000_000) * 0.90
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

        print(f"[SUCCESS] {filename} | E:{len(entities)} A:{sum(len(v) for v in attributes.values() if isinstance(v, list))} R:{len(relationships)} | Time: {processing_time}s")
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
