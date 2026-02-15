import torch


def get_loss(model, ds_loader, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in ds_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            total_loss += output.loss.item()
    
    model.train()
    return total_loss / len(ds_loader)


def generate_sample(model, ds_loader, tokenizer, device):
    model.eval()
    
    # Take one sample and decode it
    batch = next(iter(ds_loader))
    input_ids = batch["input_ids"][0]
    full_text = tokenizer.decode(input_ids, skip_special_tokens=True)

    # Encode sample
    prompt = full_text.split("### Response:")[0] + "### Response:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Inference
    model.config.use_cache = True
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
    model.config.use_cache = False

    # Decode output
    decoded_output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    response = decoded_output.split("### Response:")[1]

    # Print output
    print("\n" + "="*50)
    print("Validation generated output:")
    print(response.strip())
    print("\n" + "="*50)

    model.train()


if __name__ == '__main__':
    pass