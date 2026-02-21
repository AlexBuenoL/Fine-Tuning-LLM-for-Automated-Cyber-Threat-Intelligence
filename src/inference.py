import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from config import MODEL_NAME


MODEL_PATH = "models/best_cti_finetuned"


def preprocess_text(text, tokenizer, device):
    prompt = (
        f"### Instruction: Extract cyber threat entities in JSON format.\n"
        f"### Input: {text}\n"
        f"### Response: "
    )
    prompt_tokenized = tokenizer(
        prompt, 
        truncation=True, 
        max_length=512,
        return_tensors="pt"
    ).to(device)

    return prompt_tokenized


if __name__ == '__main__':
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load quantized model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa"
    )
    base_model.config.use_cache = True

    # Merge model with trained adapters
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()

    input_text = "The attacker used IP 192.168.1.50 to deliver the Emotet payload via phishing."
    inputs = preprocess_text(input_text, tokenizer, device)

    # Inference
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode output
    decoded_output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Extract generated part
    if "### Response:" in decoded_output:
        final_json = decoded_output.split("### Response:")[1].strip()
    else:
        final_json = decoded_output

    print(final_json)

    
    
    
