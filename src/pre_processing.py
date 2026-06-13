# import libraries

import warnings
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv", category=RuntimeWarning)
warnings.filterwarnings("ignore")
import ssl
from openie import StanfordOpenIE
import requests
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import re
import string
import spacy
from collections import defaultdict
from openie import StanfordOpenIE
from g4f.client import Client
from typing import List
import os
from groq import Groq

# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Then initialize the client
client = StanfordOpenIE()


# Initialize NLP models with SSL handling
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model for spaCy...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Configure SSL context
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Initialize StanfordOpenIE client
try:
    client = StanfordOpenIE(
        properties={
            'openie.affinity_probability_cap': 2/3,
        },
        ssl_context=ssl_context
    )
    print("StanfordOpenIE client initialized successfully.")
except Exception as e:
    print(f"Failed to initialize StanfordOpenIE: {e}")
    print("Make sure Stanford CoreNLP server is running and accessible.")
    client = None

# Check if the client was initialized
if client is None:
    print("StanfordOpenIE client is not available.")

# Step 1: Text Preprocessing
def preprocess_text(text):
    """Normalize text by lowercasing, removing stopwords and punctuation."""
    if not text or not isinstance(text, str):
        return ""

    stop_words = set(spacy.lang.en.stop_words.STOP_WORDS)
    doc = nlp(text.lower())

    processed_tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]

    return " ".join(processed_tokens)



def read_data_and_preprocess(file_path):
    """Reads data from a file and preprocesses it."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
        print(f"Raw data: {data[:500]}...") # Print first 500 characters
        print(f"Length: {len(data)}")
        processed_data = preprocess_text(data)
        logging.info(f"Processed data: {processed_data[:500]}...") # Log first 500 characters
        print(f"Processed_data: {processed_data[:500]}...") # Print first 500 characters
        return processed_data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        logging.error(f"FileNotFoundError: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading or preprocessing the file: {e}")
        logging.error(f"Error reading or preprocessing file {file_path}: {e}")
        return None

import csv
from typing import List

def read_csv_and_preprocess(file_path: str, description_col: str = "description") -> List[str]:
    """
    Read data from a CSV file, extract the description column, and preprocess each row.
    
    Args:
        file_path (str): path to the CSV file
        description_col (str): name of the description column
        
    Returns:
        List[str]: list of preprocessed descriptions
    """
    processed_descriptions = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                description = row.get(description_col, "")
                if description:
                    processed = preprocess_text(description)
                    processed_descriptions.append(processed)
                else:
                    processed_descriptions.append("")  # empty if no description
                # Optional: debug/log first few rows
                if i < 3:
                    print(f"Raw description {i}: {description[:100]}...")
                    print(f"Processed description {i}: {processed[:100]}...")
        print(f"Processed {len(processed_descriptions)} rows from CSV.")
        return processed_descriptions
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred while reading CSV: {e}")
        return []

import csv
import time

def build_text_description(row: dict) -> str:
    """
    Build a single textual description from multiple CSV fields
    to be used as LLM input.
    """
    parts = []

    def add(label, value):
        if value and isinstance(value, str) and value.strip():
            parts.append(f"{label}: {value.strip()}")

    add("This is a description about the dataset: ", row.get("title"))
    add("This dataset has subtitle: ", row.get("subtitle"))
    add("And Description: ", row.get("description"))
    add("Some Keywords about the dataset: ", row.get("keywords"))
    add("File types", row.get("fileTypes"))

    return "\n".join(parts)

def process_csv_with_llm(input_csv: str, output_file: str, llm_client, description_col: str = "description"):
    """
    Read CSV row by row, preprocess description, call LLM for extraction, write results to file.
    Pause 60s after each row.
    
    Args:
        input_csv (str): path to the CSV file
        output_file (str): path to the output file
        llm_client: client to call LLM, must have a call_llm(text) -> str method
        description_col (str): description column in the CSV
    """
    try:
        with open(input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                description = row.get(description_col, "")
                if not description:
                    print(f"Row {i}: no description, skip.")
                    continue

                # Preprocess
                processed_desc = preprocess_text(description)
                print(f"Row {i} processed description: {processed_desc[:100]}...")  # Debug

                # Call LLM
                try:
                    llm_result = llm_client.call_llm(processed_desc)  # assuming call_llm is available
                except Exception as e:
                    print(f"Error calling LLM on row {i}: {e}")
                    llm_result = "ERROR"

                # Write result to file
                with open(output_file, "a", encoding="utf-8") as out_f:
                    out_f.write(f"Row {i}:\n{llm_result}\n\n")

                print(f"Row {i} done, waiting 60s...")
                time.sleep(60)

        print("Finished processing all rows.")
    except FileNotFoundError:
        print(f"Input CSV file not found: {input_csv}")
    except Exception as e:
        print(f"An error occurred: {e}")
