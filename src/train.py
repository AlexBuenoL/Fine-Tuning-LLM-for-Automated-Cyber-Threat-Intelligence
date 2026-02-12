import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from config import MODEL_NAME, PREPROCESSED_DATA_PATH
from collators import CTICompletionColator


# Hyperparameters definition
EPOCHS = 10
LR = 5e-5


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
        batch_size=16,
        shuffle=True,
        collate_fn=data_collator
    )
    val_loader = DataLoader(
        ds["validation"],
        batch_size=16,
        shuffle=False,
        collate_fn=data_collator
    )
    test_loader = DataLoader(
        ds["test"],
        batch_size=16,
        shuffle=False,
        collate_fn=data_collator
    )

    # Model definition
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)

    # Optimizer definition
    optimizer = AdamW(model.parameters(), lr=LR)

    # LR scheduler definition
    num_training_steps = EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Training loop
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for i in range(EPOCHS):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train(device=device)