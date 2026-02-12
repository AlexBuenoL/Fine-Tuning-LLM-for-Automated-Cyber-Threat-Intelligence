from datasets import load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from config import MODEL_NAME, PREPROCESSED_DATA_PATH
from collators import CTICompletionColator


def train():
    # Load preprocessed dataset from disk
    ds = load_from_disk(PREPROCESSED_DATA_PATH)

    # Load tokenizer    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Data collator: Ensure the model only learns to generate the JSON. During training:
    # - Before the template, it replaces every tokenID of Instruction and Input with -100 
    #   --> Model weights are not updated
    # - After the template, the model uses the actual tokenIDs
    #   --> Model is trained only on predicting the JSON response
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


if __name__ == '__main__':
    train()