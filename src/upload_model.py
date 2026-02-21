import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from config import REPO_ID, MODEL_PATH, MODEL_NAME
from transformers import AutoTokenizer

def upload_to_huggingface():
    load_dotenv()
    hf_token = os.getenv("HF_WRITE_TOKEN")

    if not hf_token:
        raise ValueError("Not HS TOKEN was found.")

    login(token=hf_token)

    api = HfApi()

    try:
        api.upload_folder(
            folder_path=MODEL_PATH,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Initial upload of QLoRA adapters (CTI Fine-Tuning)"
        )
        print("Model successfully uploaded.")

        # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = "right"
        
        # tokenizer.push_to_hub(
        #     repo_id=REPO_ID,
        #     commit_message="Upload tokenizer and special tokens"
        # )

    except Exception as e:
        print(f"An error raised during the uploading: {e}")


if __name__ == '__main__':
    upload_to_huggingface()