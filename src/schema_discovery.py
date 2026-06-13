import os
import json
import zipfile
from genson import SchemaBuilder

"""

ZIP_FILE = "dataset/Hospital_Json.zip"
OUTPUT_FOLDER = "Hospital_Json_extracted"

# 1. Extract ZIP
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
    zip_ref.extractall(OUTPUT_FOLDER)

print(f"ZIP extracted to: {OUTPUT_FOLDER}\n")

# 2. Find the “root” folders in the ZIP
root_folders = set()
for root, dirs, files in os.walk(OUTPUT_FOLDER):
    for d in dirs:
        rel_path = os.path.relpath(os.path.join(root, d), OUTPUT_FOLDER)
        root_folders.add(rel_path.split(os.sep)[0])

print(f"Root datasets found: {root_folders}\n")

# 3. Iterate over each root folder
for root_name in root_folders:
    folder_path = os.path.join(OUTPUT_FOLDER, root_name)
    print(f"==============================")
    print(f"Processing dataset: {root_name}")
    print(f"Folder path: {folder_path}\n")

    all_docs = []
    merged_files = []

    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(".json"):
                file_path = os.path.join(root, f)
                merged_files.append(file_path)
                try:
                    with open(file_path, "r", encoding="utf-8") as f_in:
                        data = json.load(f_in)
                    if isinstance(data, list):
                        all_docs.extend(data)
                    elif isinstance(data, dict):
                        all_docs.append(data)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print(f"Files merged ({len(merged_files)}):")
    for f in merged_files:
        print(f"  - {f}")

    print(f"📦 Total documents after merge: {len(all_docs)}")

    # 4. Generate schema
    builder = SchemaBuilder()
    for doc in all_docs:
        builder.add_object(doc)

    schema = builder.to_schema()
    print(f"Schema generated for dataset '{root_name}':")
    print(json.dumps(schema, indent=2, ensure_ascii=False))
    print("\n\n")
"""

import zipfile
import json
import os
from collections import defaultdict
from genson import SchemaBuilder

# -------------------------
# Input
# -------------------------
zip_file = "dataset/Hospital_Json.zip"  # path to your zip file

# -------------------------
# 1. Open zip and list files
# -------------------------
with zipfile.ZipFile(zip_file, 'r') as z:
    all_files = [f for f in z.namelist() if f.lower().endswith(".json")]
    print(f"All JSON files in ZIP ({len(all_files)} files):")
    for f in all_files:
        print(f" - {f}")

# -------------------------
# 2. Group files by root folder
# -------------------------
groups = defaultdict(list)
for f in all_files:
    parts = f.split("/")
    if len(parts) > 1:
        root = parts[0]
    else:
        root = "__root__"
    groups[root].append(f)

print("\nGrouping files by root folder:")
for root, files in groups.items():
    print(f"Root: {root} -> {len(files)} file(s)")

# -------------------------
# 3. Read and merge JSON by root
# -------------------------
for root, files in groups.items():
    merged_objects = []

    print(f"\n==============================")
    print(f"📁 Root folder: {root}")
    print(f"Files to merge:")
    for f in files:
        print(f" - {f}")
        with zipfile.ZipFile(zip_file, 'r') as z:
            with z.open(f) as file:
                try:
                    data = json.load(file)
                    if isinstance(data, list):
                        merged_objects.extend(data)
                    elif isinstance(data, dict):
                        # flatten dict: if a key is a list, add to merged_objects
                        for key, val in data.items():
                            if isinstance(val, list):
                                merged_objects.extend(val)
                            else:
                                merged_objects.append({key: val})
                    else:
                        # if data is not a list or dict, skip
                        continue
                except Exception as e:
                    print(f"Error reading file {f}: {e}")

    print(f"Total objects after merge: {len(merged_objects)}")

    # -------------------------
    # 4. Generate schema with Genson
    # -------------------------
    if merged_objects:
        builder = SchemaBuilder()
        for obj in merged_objects:
            builder.add_object(obj)
        schema = builder.to_schema()
        print(f"Schema for root '{root}':")
        print(json.dumps(schema, indent=2, ensure_ascii=False))
    else:
        print(f"No valid data to generate schema for root '{root}'")
