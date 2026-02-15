import torch
import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_scheduler,
    BitsAndBytesConfig
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm.auto import tqdm
from config import MODEL_NAME, PREPROCESSED_DATA_PATH
from collators import CTICompletionColator
from dotenv import load_dotenv
from huggingface_hub import login


# Hyperparameters definition
EPOCHS = 3
LR = 2e-4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
EVAL_STEPS = 50

LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            total_loss += output.loss.item()
    
    model.train()
    return total_loss / len(val_loader)


def train(device):
    # Load preprocessed dataset from disk
    ds = load_from_disk(PREPROCESSED_DATA_PATH)

    # Load tokenizer    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Data collator: Ensure the model only learns to generate the JSON
    data_collator = CTICompletionColator(
        response_template="### Response:",
        tokenizer=tokenizer
    )

    # Build Data Loaders
    train_loader = DataLoader(
        ds["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator
    )
    val_loader = DataLoader(
        ds["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator
    )

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Model definition
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,           
        low_cpu_mem_usage=True,      
        attn_implementation="sdpa"    
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", 
                        "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Optimizer definition
    optimizer = AdamW(model.parameters(), lr=LR)

    # LR scheduler definition
    total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * EPOCHS
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    # Training loop
    progress_bar = tqdm(range(total_steps))
    model.train()
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\nStarting epoch {epoch+1}...")
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if (step+1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                progress_bar.update(1)

                if global_step % EVAL_STEPS == 0:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"[Step: {global_step}]: Train Loss: {loss.item():.2f} || Val Loss: {val_loss:.2f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model.save_pretrained("./models/best_cti_finetuned")
                        print("New best model saved.")
                    
            
if __name__ == '__main__':
    # HG logging
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(hf_token)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train(device=device)