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
from src.config import GROQ_API_KEYS, GPT_MODEL

# -------------------------
# Configuration & Paths
# -------------------------
START_FILE_INDEX = 251
END_FILE_INDEX = 500

INPUT_FOLDER = "dataset/Datasets/Full-Dataset/input"
OUTPUT_FOLDER = "output/generation/Text-To-ERD/gpt"
COST_OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "cost_report_text-to-erd_gpt.txt")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------
# Setup LLM Client & Token Tracking
# -------------------------
client = MultiKeyGroqManager(api_keys=GROQ_API_KEYS)
gpt = GPT_MODEL

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
# Prompt Templates
# -------------------------
_example_input = """
A system for registering people and their birthplaces requires a database that stores information about individuals and their respective places of birth. The system should allow the registration of people and birthplaces, as well as efficiently link individuals to their birthplaces. This system must meet the following requirements:

    * Each birthplace can have multiple people associated with it, but each person can only be linked to one birthplace.

"""

_example_output = """
{
    "tables": {
        "Person": {
            "*name": "varchar(100) NOT NULL",
            "height": "decimal(10,2)NOT NULL",
            "weight": "int()",
            "birthDate": "date()NOT NULL"
        },
        "BirthPlace": {
            "*id_birthplace": "int() NOT NULL",
            "birthCity": "varchar(100)",
            "birthState": "varchar(100)",
            "birthCountry": "varchar(100) NOT NULL",
            "+personName": "varchar(100)"
        }
    },
    "relations": [
        "BirthPlace:personName 1--* Person:name"
    ],
    "rankAdjustments": "",
    "label": ""
}
"""

_additional_example_input = """
    Consider a site where it's possible to share book reviews. As a user of this site, you can register on the platform, add books to your profile, write and share reviews, rate books, and track your reading progress through the reading status.
    """

_additional_example_output = """
{
  "tables": {
    "User": {
      "*username": "varchar(100) NOT NULL",
      "first_name": "varchar(100)",
      "last_name": "varchar(100)",
      "birth_date": "date()",
      "password": "varchar(100) NOT NULL",
      "profile_picture": "varchar(500)"
    },
    "Book": {
      "*isbn": "varchar(100) NOT NULL",
      "rating": "float() NOT NULL"
    },
    "Review": {
      "*review_id": "bigserial() NOT NULL",
      "+user_username": "varchar(100)",
      "+book_isbn": "varchar(100)",
      "start_date": "date()",
      "end_date": "date()",
      "rate": "integer()",
      "content": "text() NOT NULL"
    },
    "UserBook": {
      "*user_book_id": "bigserial() NOT NULL",
      "+user_username": "varchar(100)",
      "+book_isbn": "varchar(100)",
      "+review_id": "bigint()",
      "status": "varchar(30)"
    }
  },
  "relations": [
    "User:username 1--* Review:user_username",
    "Book:isbn 1--* Review:book_isbn",
    "User:username 1--* UserBook:user_username",
    "Book:isbn 1--* UserBook:book_isbn",
    "Review:review_id 1--* UserBook:review_id"
  ],
  "rankAdjustments": "",
  "label": "book platform"
}
"""

_json_fmt = """
{
    "tables": {
        "TableName": {
            "*PrimaryKey": "DataType",
            "+ForeignKey": "DataType",
            "Attribute": "DataType"
        }
    },
    "relations": [
        "TableOne:PrimaryKey 1--* TableTwo:ForeignKey"
    ],
    "rankAdjustments": "...",
    "label": "..."
}
"""

SYSTEM_MESSAGE = f"""
You are a database expert specializing in designing and modeling Entity-Relationship (ER) diagrams. Your task is to analyze a given system description, extract its requirements, and generate a structured ER diagram in JSON format that defines the database model.

### Requirements:

1. **Tables and Attributes**
   - Each table must have a **primary key (marked with `*`)**.
   - Foreign keys must be marked with `+` and reference another table.
   - Attributes should have appropriate data types (`varchar(200)`, `int()`, `date()`, etc.).

2. **Relationships**
   - Define entity relationships in the format:
     ```
     "TableOne:PrimaryKey 1--* TableTwo:ForeignKey"
     ```
   - The left side represents the referencing table.
   - The right side represents the referenced table.
   - Use proper cardinality indicators:
     - `?` → 0 or 1
     - `1` → Exactly 1
     - `*` → 0 or more
     - `+` → 1 or more

### Output Format:

Your response **must** strictly follow this JSON structure:

{_json_fmt}

### Example:

#### **Input:**
{_example_input}

#### **Expected Output:**
{_example_output}

 ENSURE THAT:
    - All relations explicitly specify primary and foreign keys in the format Table:PrimaryKey X--X Table:ForeignKey. Even the spaces must be respected
    - Obeserve that in a relation, Table:PrimaryKey there is no space between, you MUST obey this rule DO NOT make the relation like this: [Table: PrimaryKey]
    - rankAdjustments is always an empty list []. "
    - label contains a meaningful title."
    - The output must be in language present on example. If the txt is in english, the output is in english, but if the input is in portuguese, the output MUST be in portuguese
    "Output only the JSON—no additional text, explanations, or comments."
"""


def er_prompt(processed_text: str) -> str:
    return f"""
    Here's an additional example to help understand different relationship types:

    #### **Input:**
    {_additional_example_input}

    #### **Expected Output:**
    {_additional_example_output}

    Now, analyze the following database carefully, extract the requirements, identify tables, keys, and columns and relationships. then generate the ER diagram. Finally, translate it into the required JSON format.

    You should think about:
      What data needs to be stored in the database? (e.g., users, books, orders)
      What has distinct attributes that need to be recorded?
      What has an independent existence in the system? (e.g., a "User" can exist without an "Order," but an "Order Item" cannot exist without an "Order")
      Is the data type appropriate? (e.g., "name" should be varchar, "price" should be decimal).
      Are there constraints? (NOT NULL, UNIQUE, DEFAULT, etc.).
      How would a relationship between tables work?

    To correctly define relationships, follow this process:

      Identify dependencies: Does an object depend on another to exist? (e.g., a "Comment" depends on a "User" and a "Post")
      Determine cardinality: For each relationship, ask:
      How many elements can be associated with another?
      Can it be zero, one, or many?
      Check if an intermediary table is needed: If there's a "many-to-many" (*--*) relationship, an intermediary entity is required (e.g., "User" and "Book" need "UserBook" to track reading progress).

    To solve this problem, follow these steps:
    1. Identify the main entities from the description
    2. Determine attributes for each entity
    3. Establish primary keys for each entity
    4. Identify relationships between entities
    5. Determine foreign keys based on relationships
    6. Format everything according to the required JSON structure

    the output json must be inside '''
    Database description: {processed_text}
    """

# -------------------------
# Map generation.py format → original format
# -------------------------
def _parse_cardinality(card_str: str) -> str:
    mapping = {
        "1--1": "1:1", "1--*": "1:N", "*--1": "N:1", "*--*": "N:M",
        "1--+": "1:N", "+--1": "N:1", "?--*": "0:N", "1--?": "1:0",
        "?--1": "0:1", "+--*": "N:M",
    }
    return mapping.get(card_str, card_str)


def convert_to_original_format(er_json: dict) -> dict:
    tables = er_json.get("tables", {})
    relations = er_json.get("relations", [])

    entities = list(tables.keys())

    attributes = {}
    for table_name, cols in tables.items():
        attributes[table_name] = [k.lstrip("*+") for k in cols.keys()]

    relationships = []
    for rel in relations:
        parts = rel.split()
        if len(parts) == 3:
            left, card_str, right = parts
            entity_1 = left.split(":")[0]
            entity_2 = right.split(":")[0]
            relationships.append({
                "entity_1": entity_1,
                "entity_2": entity_2,
                "cardinality": _parse_cardinality(card_str)
            })

    return {"entities": entities, "attributes": attributes, "relationships": relationships}


# -------------------------
# Improved JSON extraction
# -------------------------
def extract_json_from_text(text: str) -> dict:
    if not text:
        return {}
    try:
        matches = re.findall(r"(?:'''|```)(?:json)?\s*(.*?)\s*(?:'''|```)", text, re.DOTALL)
        if matches:
            return json.loads(matches[-1].strip())
    except Exception:
        pass
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

        if hasattr(client, 'create_chat_completion'):
            response = client.create_chat_completion(**kwargs)
        else:
            response = client.chat.completions.create(**kwargs)

        response_content = response.choices[0].message.content

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
        prompt = er_prompt(processed_data)

        llm_output = call_llm_with_logging_tracking(
            task_name="generate_schema_from_text",
            system_content=SYSTEM_MESSAGE,
            client=client,
            model=gpt,
            user_prompt=prompt
        )

        if not llm_output:
            print(f"Warning: No output generated for {filename}. Skipping.")
            return False

        # SAFE JSON PARSE
        er_model_llm = extract_json_from_text(llm_output)
        converted = convert_to_original_format(er_model_llm)

        entities = converted["entities"]
        attributes = converted["attributes"]
        relationships = converted["relationships"]

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
