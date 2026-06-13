import json
import os

# 1. Configuration for file names and directories
input_data_file = 'dataset/Datasets/Full-Dataset/annotation.jsonl'  
folder_input = 'dataset/Datasets/Full-Dataset/input'
folder_reference = 'dataset/Datasets/Full-Dataset/Reference'

# Create directories if they don't exist
os.makedirs(folder_input, exist_ok=True)
os.makedirs(folder_reference, exist_ok=True)

def transform_answer(old_answer):
    """
    Transforms the old JSON structure into the new required format:
    - entity: List of table names
    - attribut: Dictionary mapping entities to their attributes
    - relationship: List of relationships inferred from foreign keys
    """
    new_format = {
        "entity": [],
        "attribut": {},
        "relationship": []
    }
    
    # Iterate through each table (entity) in the original answer
    for entity_name, details in old_answer.items():
        # Add to entity list
        new_format["entity"].append(entity_name)
        
        # Map attributes
        new_format["attribut"][entity_name] = details.get("Attributes", [])
        
        # Handle Foreign keys to define relationships
        foreign_keys = details.get("Foreign key", {})
        for fk_field, target_info in foreign_keys.items():
            # target_info usually looks like {"TargetEntity": "TargetField"}
            for target_entity in target_info.keys():
                rel = {
                    "entity_1": target_entity,
                    "entity_2": entity_name,
                    "cardinality": "1:N"
                }
                # Ensure duplicate relationships are not added
                if rel not in new_format["relationship"]:
                    new_format["relationship"].append(rel)
                    
    return new_format

# 2. Main execution: Read data and generate files
try:
    with open(input_data_file, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
                
            data = json.loads(line)
            
            # Save question content to input folder
            question_filename = f"{index}.txt"
            with open(os.path.join(folder_input, question_filename), 'w', encoding='utf-8') as f_q:
                f_q.write(data['question'])
            
            # Transform and save customized answer to Reference folder
            transformed = transform_answer(data['answer'])
            ref_filename = f"exercise{index}-baseline.txt"
            with open(os.path.join(folder_reference, ref_filename), 'w', encoding='utf-8') as f_r:
                json.dump(transformed, f_r, indent=2, ensure_ascii=False)

    print(f"Success! Files created in '{folder_input}' and '{folder_reference}' directories.")

except FileNotFoundError:
    print(f"Error: File '{input_data_file}' not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")