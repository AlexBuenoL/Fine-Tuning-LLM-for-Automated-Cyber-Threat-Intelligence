import json
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from config import DATASET_NAME, MODEL_NAME, PREPROCESSED_DATA_PATH


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
    tokenized_datasets.set_format("torch")

    return tokenized_datasets


if __name__ == '__main__':
    # Load dataset
    ds = load_dataset(DATASET_NAME)

    # Split into train, validation, and test (70/15/15)
    raw_datasets = split_dataset(ds)

    # Tokenize datasets
    tokenized_datasets = tokenize_datasets(raw_datasets, MODEL_NAME)

    # Save to disk
    tokenized_datasets.save_to_disk(PREPROCESSED_DATA_PATH)
    print("Preprocessed and tokenized dataset saved to " + PREPROCESSED_DATA_PATH)

    
