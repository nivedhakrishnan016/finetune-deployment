!pip install -q transformers peft accelerate trl bitsandbytes datasets

import pandas as pd
import torch
import os
import numpy as np
from google.colab import drive
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- 1. SETUP AND CONFIGURATION ---
print("🚀 Starting Fine-Tuning Setup...")

# Mount Google Drive
try:
    drive.mount('/content/drive')
except Exception as e:
    print(f"Drive mounting issue: {e}. Assuming drive is already mounted.")
    pass

# Define Paths based on your project structure
BASE_DIR = "drive/MyDrive/LLM_Project"
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data/train.csv")  # Using your prepared train data
TEST_DATA_PATH = os.path.join(BASE_DIR, "data/test.csv")    # For size check only, ensures separation
OUTPUT_DIR = os.path.join(BASE_DIR, "fine_tune_output/lora_adapter") # Final output path

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LABEL_COLUMN = "label"
TEXT_COLUMN = "text"

# Hyperparameters for fast, efficient QLoRA on 7B model
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-4

# --- 2. DATA PREPARATION ---

def create_prompt_template(sample):
    """
    Crucial formatting to force the LLM to output ONLY the label.
    """
    instruction = (
        f"Classify the user's request into one of the following labels: "
        f"billing_issue, tech_support, refund_request, shipping_delay, product_question, account_access.\n\n"
        f"Input: {sample[TEXT_COLUMN]}\n\n"
        f"Output: "
    )
    # The target output is appended to the instruction
    return instruction + f"{sample[LABEL_COLUMN]}"

# Load and process training data
try:
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"Data loaded: Train size={len(train_df)}, Test size={len(test_df)} (Held-out).")
except FileNotFoundError:
    print("FATAL: train.csv or test.csv not found. Check paths in Google Drive.")
    exit()

train_df['prompt_text'] = train_df.apply(create_prompt_template, axis=1)

# Split 95% of the loaded train_df for actual training, 5% for monitoring eval loss
# The .swapaxes warning is harmless but we keep the same splitting logic
train_subset_df, eval_subset_df = np.split(train_df.sample(frac=1, random_state=42), [int(.95 * len(train_df))])
train_dataset = Dataset.from_pandas(train_subset_df)
eval_dataset = Dataset.from_pandas(eval_subset_df)

# --- 3. MODEL AND TOKENIZER SETUP (QLoRA) ---

# 4-bit Quantization Config (NF4 is standard for QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Ideal for v5e-1 TPU for speed/stability
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False
# Gradient checkpointing and k-bit preparation for memory efficiency
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Important for training Causal LMs

# --- 4. PEFT (LoRA) CONFIGURATION ---

peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


# --- 5. TRAINING ARGUMENTS (TPU Ready - Minimal/Safe Configuration) ---

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4, # Virtual batch size is BATCH_SIZE * 4 = 32
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_strategy="epoch", # Save checkpoint after each epoch
    learning_rate=LR,
    bf16=True, # Use bfloat16 for TPU v5e-1
    group_by_length=True,
    disable_tqdm=False,
    # ALL evaluation arguments removed to prevent TypeError:
    # evaluation_strategy, eval_steps, load_best_model_at_end, metric_for_best_model
)

# --- 6. INITIALIZE AND TRAIN SFTTRAINER (ABSOLUTELY FINAL, MINIMALIST VERSION) ---

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=create_prompt_template,
    args=training_args,
    # REMOVED: max_seq_length=512,
    # REMOVED: dataset_text_field="prompt_text", <-- THIS CAUSED THE LATEST TypeError
)

print("\n\n🔥 Starting Fine-Tuning (3 Epochs)...")
# Training should finally start now.
trainer.train()

# --- 7. SAVE FINAL ADAPTER WEIGHTS ---
final_output_path = OUTPUT_DIR
# Ensure the tokenizer (config files) and the trained adapter weights are saved
tokenizer.save_pretrained(final_output_path)
trainer.model.save_pretrained(final_output_path)

print(f"\n\n✅ Fine-tuning complete. Adapter weights saved to: {final_output_path}")

# --- NEXT STEPS HINT ---
print("\nNEXT STEPS: 1. Download the lora_adapter folder from Google Drive. 2. Place it in docker/adapter. 3. Deploy to AWS.")[