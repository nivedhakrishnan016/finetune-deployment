import time
import uuid
import logging
import os
import json
import math
from functools import lru_cache
from typing import List, Dict

from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

import torch
# Using AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel # Required for LoRA/QLoRA inference

# ====== Config ======
# NOTE: ADAPTER_PATH is correctly set to use the Docker path /app/lora_adapter
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "/app/lora_adapter")
MODEL_NAME_OR_PATH = os.environ.get("MODEL_NAME_OR_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
# FIX 1: Force DEVICE to 'cpu' when running on CPU-only instance (c6i.xlarge)
DEVICE = "cpu"

# Match the labels you used in your fine-tuning prompt
LABELS = ["billing_issue", "tech_support", "refund_request", "shipping_delay", "product_question", "account_access"]

# Logging (structured for observability)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("inference_service")

app = Flask(__name__)

def make_prompt(text: str) -> str:
    """Creates the exact prompt format used during SFT training."""
    instruction = (
        f"Classify the user's request into one of the following labels: "
        f"billing_issue, tech_support, refund_request, shipping_delay, product_question, account_access.\n\n"
        f"Input: {text.strip()}\n\n"
        f"Output: "
    )
    return instruction

@lru_cache(maxsize=1)
def load_model_and_tokenizer():
    """Loads base model, tokenizer, and merges LoRA adapter."""
    # FIX 2: Log the fixed DEVICE
    logger.info(json.dumps({"event": "model_load_start", "device": DEVICE, "base_model": MODEL_NAME_OR_PATH}))

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model
    # FIX 3: Removed GPU-only parameters for CPU compatibility
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        # REMOVED: load_in_8bit=True, # CPU cannot use 8bit/4bit quantization
        torch_dtype=torch.float32, # Use standard float32 for CPU
        device_map=DEVICE,         # Explicitly map to "cpu"
        low_cpu_mem_usage=True     # Keep this for memory efficiency
    )

    # 3. Load PEFT Adapter
    try:
        if os.path.isdir(ADAPTER_PATH) and len(os.listdir(ADAPTER_PATH)) > 0:
            model = PeftModel.from_pretrained(model, ADAPTER_PATH)
            logger.info(json.dumps({"event": "adapter_loaded", "adapter_path": ADAPTER_PATH, "status": "success"}))
        else:
            logger.warning(json.dumps({"event": "adapter_load_skip", "adapter_path": ADAPTER_PATH, "status": "directory_empty_or_missing"}))
    except Exception as e:
        logger.error(json.dumps({"event": "adapter_load_error", "error": str(e), "path": ADAPTER_PATH}))

    # FIX 4: Ensure model is moved to CPU device if it wasn't already (redundant but safe)
    if DEVICE == "cpu":
        model.to(DEVICE)

    model.eval()
    logger.info(json.dumps({"event": "model_load_complete", "status": "ready"}))
    return model, tokenizer

def compute_label_scores(model, tokenizer, prompt: str, labels: List[str]) -> Dict[str, float]:
    """Computes likelihood of each label given the prompt using loss scoring."""
    model_device = next(model.parameters()).device
    losses = []

    # Get the token IDs for the prompt part
    prompt_enc = tokenizer(prompt, return_tensors="pt")
    prompt_ids = prompt_enc["input_ids"][0]
    prompt_len = prompt_ids.shape[0]

    for label in labels:
        full_text = prompt + label
        enc = tokenizer(full_text, return_tensors="pt")
        # Ensure tensors are moved to the correct device (CPU)
        input_ids = enc["input_ids"].to(model_device)

        # Mask prompt tokens (-100) so loss is computed only on label tokens
        labels_ids = input_ids.clone()
        labels_ids[0, :prompt_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels_ids)
            loss = outputs.loss.item()
        losses.append(loss)

    # Convert losses -> scores/probs (Math logic remains correct)
    neg_losses = [ -l for l in losses ]
    max_neg_loss = max(neg_losses)
    exps = [ math.exp(x - max_neg_loss) for x in neg_losses ]