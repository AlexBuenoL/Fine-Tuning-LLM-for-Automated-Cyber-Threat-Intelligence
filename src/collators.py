import torch
from transformers import DataCollatorForLanguageModeling

class CTICompletionCollator(DataCollatorForLanguageModeling):
    def __init__(self, response_template, tokenizer, *args, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=False, *args, **kwargs)
        self.response_template = response_template
        self.tokenizer = tokenizer

    def torch_call(self, examples):
        batch = super().torch_call(examples)

        # Identify token IDs for "### Response:"
        response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)

        for i in range(len(batch["labels"])):
            input_ids = batch["input_ids"][i].tolist()

            # Find the slice where the actual answer starts
            response_start_idx = -1
            for idx in range(len(input_ids) - len(response_token_ids) + 1):
                if input_ids[idx : idx + len(response_token_ids)] == response_token_ids:
                    response_start_idx = idx + len(response_token_ids)
                    break
            
            # Apply the masking (-100) to the instruction part
            if response_start_idx != -1:
                batch["labels"][i, :response_start_idx] = -100
        
        return batch