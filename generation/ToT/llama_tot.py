import os
import json
import time
import sys
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Setup root directory
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pre_processing import preprocess_text
from src.setup_llm import MultiKeyGroqManager

try:
    from src.config import GROQ_API_KEYS, LLAMA_MODEL
except ImportError:
    # Fallback keys if config module doesn't exist or misses these vars
    GROQ_API_KEYS = [
        "pass your api key here"
    ]
    LLAMA_MODEL = "llama3-70b-8192"

# ======================================================
# CONFIGURATION
# ======================================================
START_FILE_INDEX = 251
END_FILE_INDEX = 500

INPUT_FOLDER = os.path.join(project_root, "dataset/Datasets/Full-Dataset/input")
OUTPUT_FOLDER = os.path.join(project_root, "output/generation/prompt_ToT_llama")
COST_OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "cost_report_prompt_tot_llama.txt")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ======================================================
# LLM SETUP & TRACKING
# ======================================================
client = MultiKeyGroqManager(api_keys=GROQ_API_KEYS)

total_prompt_tokens = 0
total_completion_tokens = 0
total_calls = 0
batch_total_tokens = 0
batch_total_cost = 0
batch_files_processed = 0
stats_lock = threading.Lock()

def extract_json_from_text(text: str) -> dict:
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    try:
        text = text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except:
        return {}

def tracked_call_llm(task_name: str, model_name: str, prompt: str) -> str:
    global total_prompt_tokens, total_completion_tokens, total_calls
    try:
        with stats_lock:
            total_calls += 1
        
        kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are an expert database designer using Tree of Thoughts."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 5000,
            "stream": False
        }
        
        if hasattr(client, 'create_chat_completion'):
            response = client.create_chat_completion(**kwargs)
        else:
            response = client.chat.completions.create(**kwargs)
        
        content = response.choices[0].message.content
        
        if hasattr(response, 'usage') and response.usage:
            with stats_lock:
                total_prompt_tokens += response.usage.prompt_tokens
                total_completion_tokens += response.usage.completion_tokens
        else:
            with stats_lock:
                total_prompt_tokens += len(prompt) // 4
                total_completion_tokens += len(content) // 4
                
        return content
    except Exception as e:
        tqdm.write(f"[{task_name}] LLM Call Error: {e}")
        return ""

# ======================================================
# PROMPT TEMPLATES (OneStep ToT Approach)
# ======================================================
def candidate_generation_prompt(processed_text: str) -> str:
    return f"""
You are a database design expert. Generate 3 DIFFERENT versions of Entity-Relationship (ER) models from the following domain description.

DOMAIN DESCRIPTION:
\"\"\"
{processed_text}
\"\"\"

INSTRUCTIONS:
1. Create 3 ER model versions with different approaches:
   - Version 1: Focus on main entities and basic relationships
   - Version 2: Focus on detailed attributes and complex relationships
   - Version 3: Balance between simplicity and completeness

2. GENERAL RULES:
   - Entity names MUST be uppercase
   - Attribute names MUST be lowercase
   - Attributes must be grouped by entity
   - Use ONLY English

3. JSON FORMAT:
{{
  "candidate_1": {{
    "entities": ["ENTITY1", "ENTITY2"],
    "attributes": {{
      "ENTITY1": ["attr1", "attr2"],
      "ENTITY2": ["attrA"]
    }},
    "relationships": [
      {{
        "entity_1": "ENTITY1",
        "entity_2": "ENTITY2",
        "cardinality": "1:1|1:N|N:M",
        "description": "Brief relationship description"
      }}
    ],
    "rationale": "Explanation for this approach"
  }},
  "candidate_2": {{ ... }},
  "candidate_3": {{ ... }}
}}

Output ONLY JSON without any additional text.
"""

def evaluation_prompt(processed_text: str, candidates: dict) -> str:
    return f"""
You are an ER model evaluation expert. Evaluate the 3 candidate ER models below based on the domain description.

DOMAIN DESCRIPTION:
\"\"\"
{processed_text}
\"\"\"

CANDIDATES:
{json.dumps(candidates, indent=2, ensure_ascii=False)}

EVALUATION CRITERIA:
1. Completeness (40%): Covers all concepts in the description
2. Accuracy (30%): Correctly reflects relationships and attributes
3. Efficiency (20%): No redundant or missing entities
4. Consistency (10%): Consistent naming, clear structure

REQUIREMENTS:
1. Score each candidate (0-100) for each criterion
2. Calculate weighted total score
3. Select the best candidate
4. Provide feedback for each candidate

OUTPUT FORMAT (JSON):
{{
  "evaluation": {{
    "candidate_1": {{
      "scores": {{
        "completeness": 85,
        "accuracy": 90,
        "efficiency": 80,
        "consistency": 85
      }},
      "total_score": 85.5,
      "feedback": "Detailed feedback on strengths and weaknesses"
    }},
    "candidate_2": {{ ... }},
    "candidate_3": {{ ... }}
  }},
  "best_candidate": "candidate_1",
  "final_model": {{entities, attributes, relationships from best candidate}},
  "improvements": "Suggested improvements for final model"
}}

Output ONLY JSON without any additional text.
"""

def refinement_prompt(processed_text: str, best_model: dict, evaluation: dict) -> str:
    return f"""
You are an ER model refinement expert. Improve the model based on the evaluation.

DOMAIN DESCRIPTION:
\"\"\"
{processed_text}
\"\"\"

CURRENT MODEL (best candidate):
{json.dumps(best_model, indent=2, ensure_ascii=False)}

EVALUATION AND FEEDBACK:
{json.dumps(evaluation, indent=2, ensure_ascii=False)}

REQUIREMENTS:
1. Integrate improvement suggestions from the evaluation
2. Fix any identified issues
3. Retain strengths
4. Ensure the final model is optimal

OUTPUT FORMAT (JSON):
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
      "cardinality": "1:1|1:N|N:M",
      "description": "Brief relationship description"
    }}
  ],
  "improvements_made": "Description of improvements implemented"
}}

Output ONLY JSON without any additional text.
"""

# ======================================================
# PIPELINE LOGIC
# ======================================================
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

    try:
        tqdm.write(f"[START] Processing {filename} via Prompt ToT Pipeline (LLaMA)...")
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_description = f.read().strip()
        if not raw_description: 
            return False

        processed_data = preprocess_text(raw_description)

        # 1. Generate Candidates
        cand_prompt = candidate_generation_prompt(processed_data)
        cand_out = tracked_call_llm("Generate_Candidates", LLAMA_MODEL, cand_prompt)
        candidates = extract_json_from_text(cand_out)

        if not candidates or "candidate_1" not in candidates:
            tqdm.write(f"[ERROR] Failed to generate valid candidates for {filename}")
            return False

        # 2. Evaluate Candidates
        eval_prompt = evaluation_prompt(processed_data, candidates)
        eval_out = tracked_call_llm("Evaluate_Candidates", LLAMA_MODEL, eval_prompt)
        evaluation = extract_json_from_text(eval_out)

        best_candidate_key = evaluation.get("best_candidate", "candidate_1")
        best_model = evaluation.get("final_model", candidates.get(best_candidate_key, list(candidates.values())[0]))

        # 3. Refine Best Model
        refine_prompt = refinement_prompt(processed_data, best_model, evaluation)
        refine_out = tracked_call_llm("Refine_Model", LLAMA_MODEL, refine_prompt)
        final_model = extract_json_from_text(refine_out)

        if not final_model:
            final_model = best_model

        # Ensure standard structure expected for ER schema downstream
        entities = final_model.get("entities", final_model.get("entity", []))
        attributes = final_model.get("attributes", final_model.get("attribute", {}))
        relationships = final_model.get("relationships", final_model.get("relationship", []))

        processing_time = round(time.time() - start_time, 2)

        result_dict = {
            "filename": filename,
            "entity": entities,
            "attribute": attributes,
            "relationship": relationships,
            "processing_time": processing_time,
            "metadata": {
                "best_candidate": best_candidate_key,
                "improvements_made": final_model.get("improvements_made", "None")
            }
        }
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(result_dict, f_out, ensure_ascii=False, indent=2)

        with stats_lock:
            p_toks = total_prompt_tokens - file_start_prompt
            c_toks = total_completion_tokens - file_start_completion
            file_cost = ((p_toks / 1_000_000) * 0.80) + ((c_toks / 1_000_000) * 0.90)
            batch_total_tokens += (p_toks + c_toks)
            batch_total_cost += file_cost
            batch_files_processed += 1

        tqdm.write(f"[SUCCESS] {filename} | E:{len(entities)} A:{sum(len(v) for v in attributes.values() if isinstance(v, list))} R:{len(relationships)} | Time: {processing_time}s | Cost: ${file_cost:.4f}")
        return True

    except Exception as e:
        tqdm.write(f"ERROR processing {filename}: {e}")
        return False

# ======================================================
# EXECUTION
# ======================================================
if __name__ == "__main__":
    input_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.txt')]
    input_files = [f for f in input_files if (nums := re.findall(r'\d+', f)) and START_FILE_INDEX <= int(nums[-1]) <= END_FILE_INDEX]
    input_files.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
    
    print(f"\nFOUND {len(input_files)} FILES TO PROCESS WITH PROMPT ToT PIPELINE (LLaMA)...")
    
    # Process files concurrently (using threads since mostly waiting for API)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_single_file, filename): filename for filename in input_files}
        for future in tqdm(as_completed(futures), total=len(input_files), desc="ToT Pipeline LLaMA", unit="file"):
            future.result()

    with open(COST_OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"--- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"Files Processed: {batch_files_processed} | Tokens: {batch_total_tokens} | Cost: ${batch_total_cost:.6f}\n\n")
    print(f"\nCOMPLETED. Log saved at: {COST_OUTPUT_FILE}")
