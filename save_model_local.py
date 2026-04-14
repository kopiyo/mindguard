"""
Run this ONCE in PowerShell:
    python save_model_local.py

It downloads mental/mental-roberta-base and saves it locally.
After this, the Streamlit app will load in seconds.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, json, os

HF_TOKEN = input("Paste your HuggingFace token and press Enter: ").strip()

print("\nStep 1/4: Loading config...")
with open("mindguard_model_config.json") as f:
    config = json.load(f)
model_name = config["model_name"]
print(f"         Model: {model_name}")

print("Step 2/4: Downloading and saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN,
)
tokenizer.save_pretrained("mindguard_tokenizer")
print("         Tokenizer saved to ./mindguard_tokenizer")

print("Step 3/4: Downloading base model architecture...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    ignore_mismatched_sizes=True,
    token=HF_TOKEN,
)
print("         Base model downloaded.")

print("Step 4/4: Loading your trained weights and saving model locally...")
device = torch.device("cpu")
state_dict = torch.load("mindguard_best_weights.pt", map_location=device)
model.load_state_dict(state_dict)
os.makedirs("mindguard_model_local", exist_ok=True)
model.save_pretrained("mindguard_model_local")
print("         Model saved to ./mindguard_model_local")

print("\nDone! Your model is now saved locally.")
print("The Streamlit app will now load fast without downloading anything.")
print("\nRun your app with:")
print("    streamlit run Try_streamlit_app.py")
