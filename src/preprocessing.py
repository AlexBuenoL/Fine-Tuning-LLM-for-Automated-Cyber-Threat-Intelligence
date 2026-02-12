import json
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from torch.utils.data import DataLoader


DATASET_NAME = "mrmoor/cyber-threat-intelligence"
MODEL_NAME = "mistralai/Mistral-7B-v0.3"


def split_dataset(loaded_dataset):
    df = loaded_dataset["train"]
    train_testval = df.train_test_split(test_size=0.3)
    test_val = train_testval["test"].train_test_split(test_size=0.5)

    raw_datasets = DatasetDict({
        "train": train_testval["train"],
        "validation": test_val["train"],
        "test": test_val["test"]
    })

    return raw_datasets


def tokenize_datasets(raw_datasets, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # Mistral does not have pad token id
    tokenizer.padding_side = "right"

    def tokenize_function(samples):
        prompts = []
        for text, entities in zip(samples["text"], samples["entities"]):
            # Extract entities from the input text
            extracted = [{ "entity": ent["label"], "value": text[ent["start_offset"]:ent["end_offset"]] } 
                         for ent in entities]
            
            # Build prompt: Instruction + Input + Response
            prompt = (
                f"### Instruction: Extract cyber threat entities in JSON format.\n"
                f"### Input: {text}\n"
                f"### Response: {json.dumps(extracted)}{tokenizer.eos_token}"
            )
            prompts.append(prompt)
        
        return tokenizer(prompts, truncation=True, max_length=512)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(raw_datasets["train"].column_names)
    tokenized_datasets = tokenized_datasets.set_format("torch")

    # Ensure the model only learns to generate the JSON. During training:
    # - Before the template, it replaces every tokenID of Instruction and Input with -100 
    #   --> Model weights are not updated
    # - After the template, the model uses the actual tokenIDs
    #   --> Model is trained only on predicting the JSON response
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template="### Response:",
        tokenizer=tokenizer
    )

    return tokenized_datasets, data_collator


if __name__ == '__main__':
    # Load dataset
    ds = load_dataset(DATASET_NAME)

    # Split into train, validation, and test (70/15/15)
    raw_datasets = split_dataset(ds)

    # Tokenize datasets
    tokenized_datasets, data_collator = tokenize_datasets(raw_datasets, MODEL_NAME)

    # Build PyTorch Dataloaders
    train_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=16,
        shuffle=True,
        collate_fn=data_collator
    )
    val_loader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=16,
        shuffle=False,
        collate_fn=data_collator
    )
    test_loader = DataLoader(
        tokenized_datasets["test"],
        batch_size=16,
        shuffle=False,
        collate_fn=data_collator
    )