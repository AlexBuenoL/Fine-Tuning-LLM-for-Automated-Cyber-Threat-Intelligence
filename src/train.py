from transformers import AutoTokenizer, DataCollatorForCompletionOnlyLM
from config import MODEL_NAME


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Ensure the model only learns to generate the JSON. During training:
    # - Before the template, it replaces every tokenID of Instruction and Input with -100 
    #   --> Model weights are not updated
    # - After the template, the model uses the actual tokenIDs
    #   --> Model is trained only on predicting the JSON response
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template="### Response:",
        tokenizer=tokenizer
    )