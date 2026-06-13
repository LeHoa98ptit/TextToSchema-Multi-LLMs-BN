# import libraries

import warnings
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv", category=RuntimeWarning)
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
import groq
import hashlib
import time
import threading


# Create logging directory if it doesn't exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(BASE_DIR, "Logging")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create caching directory
cache_dir = os.path.join(BASE_DIR, "Cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Config logging
log_path = os.path.join(log_dir, "llm_pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

class MultiKeyGroqManager:
    """Handles automatic API key rotation when rate limits are hit."""
    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("List of API keys cannot be empty")
        self.api_keys = api_keys
        self.current_idx = 0
        self.client = Groq(api_key=self.api_keys[self.current_idx], max_retries=0)
        self.num_keys = len(api_keys)
        self.lock = threading.Lock()
        
    def switch_key(self, current_failing_key=None):
        with self.lock:
            # If another thread has already switched the key, skip to avoid skipping keys
            if current_failing_key and self.api_keys[self.current_idx] != current_failing_key:
                return
                
            self.current_idx = (self.current_idx + 1) % self.num_keys
            new_key = self.api_keys[self.current_idx]
            self.client = Groq(api_key=new_key, max_retries=0)
            masked_key = f"...{new_key[-4:]}" if len(new_key) > 4 else "..."
            logging.warning(f"Switched to next API key: {masked_key}")
        
    def create_chat_completion(self, **kwargs):
        max_attempts = self.num_keys * 4
        attempts = 0
        backoff = 2
        
        while attempts < max_attempts:
            # Save the key and client of the current thread
            current_key = self.api_keys[self.current_idx]
            current_client = self.client
            try:
                return current_client.chat.completions.create(**kwargs)
            except (groq.RateLimitError, groq.APIError) as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str:
                    logging.warning(f"RateLimit hit. Backing off {backoff}s and switching...")
                    time.sleep(backoff) # Backoff before switching
                    self.switch_key(current_failing_key=current_key)
                    attempts += 1
                    backoff = min(backoff * 2, 60) # Exponential backoff up to 60s
                elif "401" in err_str or "invalid" in err_str or "unauthorized" in err_str:
                    with self.lock:
                        if current_key in self.api_keys:
                            logging.error(f"Invalid API Key detected! Removing from pool...")
                            self.api_keys.remove(current_key)
                            self.num_keys -= 1
                            if self.num_keys == 0:
                                raise Exception("All API keys are invalid or exhausted!")
                            self.current_idx = self.current_idx % self.num_keys
                            self.client = Groq(api_key=self.api_keys[self.current_idx], max_retries=0)
                    attempts += 1
                else:
                    raise e
                
        raise Exception("All API keys hit their rate limits or max retry attempts reached.")


def call_llm_with_logging(task, content, client, model, prompt: str) -> str:
    """
    Call LM to extract and log fully
    Args:
        client: LLM client
        prompt: Prompt to send to the LLM
    Returns:
        str: The response from the LLM
    """
    
    # 1. Check cache first to save tokens and time
    prompt_hash = hashlib.md5((model + task + prompt).encode('utf-8')).hexdigest()
    cache_path = os.path.join(cache_dir, f"{prompt_hash}.json")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            logging.info(f"Cache hit for {task}! Skipping API call to save tokens.")
            return cached_data['response']
        except Exception as e:
            logging.warning(f"Failed to read cache: {e}")

    # Log before calling LLMs
    logging.info(f"Starting {task} extract with prompt: {prompt[:300]}...")  # Log 200

    try:
        kwargs = {
            "messages": [
                {"role": "system", "content": str(content)},
                {"role": "user", "content": str(prompt)}
            ],
            "model": model,
            "temperature": 0.3
        }
        
        # Call LLM
        if hasattr(client, 'create_chat_completion'):
            chat_completion = client.create_chat_completion(**kwargs)
        else:
            chat_completion = client.chat.completions.create(**kwargs)

        response_content = chat_completion.choices[0].message.content

        # Log result successfully
        logging.info(
            json.dumps({
                "event": task.upper() + "_EXTRACT_SUCCESS",
                "prompt_hash": hash(prompt),
                "response": response_content[:500],
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "usage": {
                    "input_tokens": chat_completion.usage.prompt_tokens,
                    "output_tokens": chat_completion.usage.completion_tokens
                }
            })
        )
    
        # Save to cache
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({
                "response": response_content,
                "model": model,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)

        return response_content
    except Exception as e:
        # Detailed error log
        logging.error(
            json.dumps({
                "event": task.upper() + "_EXTRACT_FAILED",
                "error_type": type(e).__name__,
                "error_msg": str(e),
                "prompt_hash": hash(prompt),
                "timestamp": datetime.now().isoformat(),
                "model": model
            })
        )
        raise  # Re-raise exception after log
